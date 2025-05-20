import numpy as np
import tensorflow as tf
from PIL import Image

def predict_large_image(model: tf.keras.Model,
                        img_path: str,
                        batch_size: int = 4) -> Image.Image:
    # 1) Get model’s expected tile size & channels
    _, tile_h, tile_w, tile_c = model.input_shape

    # 2) Read raw bytes & decode with channels=0 (preserve file’s native channels)
    raw = tf.io.read_file(img_path)
    img = tf.io.decode_image(raw,
                              channels=0,           # let TF pick 1/3/4 channels appropriately :contentReference[oaicite:0]{index=0}
                              dtype=tf.uint8)
    arr = tf.cast(img, tf.float32) / 255.0
    arr = arr.numpy()  # shape = (H_full, W_full, C_in)

    H_full, W_full, C_in = arr.shape

    # 3) Pad spatial dims so they’re multiples of tile_h/tile_w
    pad_h = (tile_h - H_full % tile_h) % tile_h
    pad_w = (tile_w - W_full % tile_w) % tile_w
    arr = np.pad(
        arr,
        ((0, pad_h), (0, pad_w), (0, max(0, tile_c - C_in))),
        mode='constant', constant_values=0            # pad channels/truncate below :contentReference[oaicite:1]{index=1}
    )
    H_pad, W_pad, _ = arr.shape

    # 4) Slice into non-overlapping tiles
    tiles, coords = [], []
    for y in range(0, H_pad, tile_h):
        for x in range(0, W_pad, tile_w):
            tile = arr[y:y+tile_h, x:x+tile_w]
            # If file had >4 channels, truncate extras
            if tile.shape[-1] > tile_c:
                tile = tile[..., :tile_c]
            tiles.append(tile)
            coords.append((y, x))
    tiles = np.stack(tiles, axis=0)

    # 5) Batch predict
    preds = []
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i+batch_size]
        p = model.predict(batch, verbose=0)
        preds.append(p)
    preds = np.concatenate(preds, axis=0)

    # 6) Stitch back
    C_out = preds.shape[-1]
    stitched = np.zeros((H_pad, W_pad, C_out), dtype=np.float32)
    for (y, x), tile_pred in zip(coords, preds):
        stitched[y:y+tile_h, x:x+tile_w] = tile_pred

    # 7) Crop to original
    stitched = stitched[:H_full, :W_full]

    # 8) Convert to PIL
    if stitched.shape[-1] == 1:
        arr_out = (np.clip(stitched[...,0], 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr_out, mode='L')
    else:
        arr_out = (np.clip(stitched, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr_out)

# Example usage
if __name__ == "__main__":
    import sys
    model = tf.keras.models.load_model(sys.argv[1])
    mask = predict_large_image(model, sys.argv[2], batch_size=8)
    mask.save("prediction_mask.png")
    print("Saved full-size prediction to prediction_mask.png")
