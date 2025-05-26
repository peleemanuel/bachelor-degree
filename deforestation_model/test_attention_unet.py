import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def predict_large_image(model: tf.keras.Model,
                        img_path: str,
                        batch_size: int = 4,
                        visualize: bool = False,
                        save_overlay: bool = False) -> Image.Image:
    """
    Tile an arbitrary-size multi-channel image into patches that match
    model.input_shape[1:], run predictions, then reassemble and return the mask.

    If visualize=True, afișează măștile și overlay-ul.
    If save_overlay=True, salvează 'overlay.png' cu original+mască.
    """

    # 1) Obține dimensiunile așteptate de model
    _, tile_h, tile_w, tile_c = model.input_shape

    # 2) Încarcă & normalizează păstrând numărul nativ de canale
    raw = tf.io.read_file(img_path)
    img_tf = tf.io.decode_image(raw, channels=0, dtype=tf.uint8)
    arr = tf.cast(img_tf, tf.float32) / 255.0
    arr = arr.numpy()                     

    H_full, W_full, C_in = arr.shape

    # 3) Padding spațial și pe canale până la tile_c
    pad_h = (tile_h - H_full % tile_h) % tile_h
    pad_w = (tile_w - W_full % tile_w) % tile_w
    arr = np.pad(arr,
                 ((0, pad_h), (0, pad_w), (0, max(0, tile_c - C_in))),
                 mode='constant', constant_values=0)  # zero‐pad canale :contentReference[oaicite:1]{index=1}
    H_pad, W_pad, _ = arr.shape

    # 4) Tăiere în tile_h × tile_w × tile_c
    tiles, coords = [], []
    for y in range(0, H_pad, tile_h):
        for x in range(0, W_pad, tile_w):
            tile = arr[y:y+tile_h, x:x+tile_w]
            if tile.shape[-1] > tile_c:
                tile = tile[..., :tile_c]
            tiles.append(tile)
            coords.append((y, x))
    tiles = np.stack(tiles, axis=0)

    # 5) Preziceri pe batch-uri
    preds = []
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i+batch_size]
        p = model.predict(batch, verbose=0)
        preds.append(p)
    preds = np.concatenate(preds, axis=0)

    # 6) Reasamblare într-un tensor full-size
    C_out = preds.shape[-1]
    stitched = np.zeros((H_pad, W_pad, C_out), dtype=np.float32)
    for (y, x), tile_pred in zip(coords, preds):
        stitched[y:y+tile_h, x:x+tile_w] = tile_pred

    # 7) Crop la dimensiunile originale
    stitched = stitched[:H_full, :W_full]

    # 8) Convertire la PIL maskă
    if C_out == 1:
        arr_out = (np.clip(stitched[...,0], 0, 1) * 255).astype(np.uint8)
        mask_img = Image.fromarray(arr_out, mode='L')
    else:
        arr_out = (np.clip(stitched, 0, 1) * 255).astype(np.uint8)
        mask_img = Image.fromarray(arr_out)

    # 9) Vizualizare opțională
    if visualize:
        orig = np.array(Image.open(img_path).convert('RGB'))
        mask_arr = np.array(mask_img)

        plt.figure(figsize=(6,6))
        plt.title("Predicted Mask")
        plt.imshow(mask_arr, cmap="gray")
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(6,6))
        plt.title("Mask Overlay")
        plt.imshow(orig)
        plt.imshow(mask_arr, cmap="jet", alpha=0.4)  # transparență 40% :contentReference[oaicite:2]{index=2}
        plt.axis("off")
        plt.show()

    # 10) Generare & salvare imagine overlay cu PIL
    if save_overlay:
        background = Image.open(img_path).convert("RGBA")
        foreground = mask_img.convert("RGBA")

        # Creează o imagine semi-transparentă roșie pentru masca binară
        red_mask = Image.new("RGBA", foreground.size, (255,0,0,0))
        # Folosește canalul L al măștii ca alfa
        red_mask.putalpha(foreground.split()[0])
        # Compune peste original
        overlay = Image.alpha_composite(background, red_mask)
        overlay.save("overlay.png")              # salvare overlay :contentReference[oaicite:3]{index=3}

    return mask_img


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <model_path> <image_path> [visualize] [save_overlay]")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    visualize   = "visualize"   in sys.argv[3:]
    save_overlay= "save_overlay" in sys.argv[3:]

    model = tf.keras.models.load_model(model_path)
    mask  = predict_large_image(model, image_path,
                                batch_size=8,
                                visualize=visualize,
                                save_overlay=save_overlay)
    mask.save("prediction_mask.png")
    print("Saved mask to prediction_mask.png")
    if save_overlay:
        print("Saved overlay image to overlay.png")
