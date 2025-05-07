import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
import geopandas as gpd
from matplotlib import cm
import time

# Adjust this import to point to your model definition
from unet_model import UNetAM
from rasterio.features import rasterize
from skimage import exposure

class PatchSequence(Sequence):
    """
    Streams 512x512 patches from a GeoTIFF and its full-scene mask (.npy) on the fly.
    Supports flexible single-band to RGB conversion via histogram stretching and colormap.
    """
    def __init__(self, img_path, mask_path, windows, batch_size=4,
                 normalize=True, colormap='viridis', stretch_percent=(2, 98), **kwargs):
        # Allow Keras to pass through workers/use_multiprocessing kwargs
        super().__init__(**kwargs)
        self.img_src = rasterio.open(img_path)
        self.mask = np.load(mask_path)
        self.windows = windows
        self.bs = batch_size
        self.norm = normalize
        self.colormap = colormap
        self.stretch_percent = stretch_percent

    def __len__(self):
        return int(np.ceil(len(self.windows) / float(self.bs)))

    def __getitem__(self, idx):
        batch = self.windows[idx * self.bs : (idx + 1) * self.bs]
        X, Y = [], []
        for row_off, col_off in batch:
            win = Window(col_off, row_off, 512, 512)

            # Read single band directly
            gray = self.img_src.read(1, window=win).astype('float32')
            if self.norm:
                gray /= 10000.0

            # Optional: Percentile stretch for contrast enhancement
            p_min, p_max = np.percentile(gray, self.stretch_percent)
            gray_stretched = exposure.rescale_intensity(
                gray, in_range=(p_min, p_max), out_range=(0, 1)
            )

            # Apply matplotlib colormap to get RGB
            cmap = cm.get_cmap(self.colormap)
            img_rgba = cmap(gray_stretched)
            img = img_rgba[..., :3].astype('float32')  # drop alpha channel

            # Extract corresponding mask patch
            m = self.mask[row_off:row_off+512, col_off:col_off+512]
            m = np.expand_dims(m, -1).astype('float32')

            X.append(img)
            Y.append(m)

        return np.stack(X, 0), np.stack(Y, 0)


def make_windows(width, height, patch_size=512, stride=512):
    """
    Generate top-left coords for non-overlapping windows.
    """
    windows = []
    for r in range(0, height, stride):
        for c in range(0, width, stride):
            if r + patch_size <= height and c + patch_size <= width:
                windows.append((r, c))
    return windows


def train_per_folder(model, img_dir, df, processed_images):

    if os.path.exists('unet_attention_weights.weights.h5'):
        print("Loading existing weights...")
        model.load_weights('unet_attention_weights.weights.h5')
    else:
        print("No existing weights found, starting fresh.")
    
    for fname in sorted(os.listdir(img_dir)):

        # Skip if already processed
        img_path = os.path.join(img_dir, fname)
        if img_path in processed_images:
            print(f"Skipping {img_path} as it has already been processed.")
            continue

        if not fname.lower().endswith('.jp2'):
            continue
        img_path  = os.path.join(img_dir, fname)
        # Create the mask path dynamically from the raster metadata
        with rasterio.open(img_path, "r", driver='JP2OpenJPEG') as src:
            H, W = src.height, src.width
            meta = src.meta
            mask_path = os.path.join(img_dir, fname.replace('.jp2', '_mask.npy'))
            if not os.path.exists(mask_path):
                print(f"Creating mask for {fname}...")
            # Transform polygons to the raster CRS
            df = df.to_crs(meta['crs'])
            
            # Create shapes from the geometries
            shapes = [(geom, 1) for geom in df.geometry]
            
            # Rasterize the shapes to create the mask
            mask = rasterize(
                shapes,
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype='uint8'
            )
            
            # Save the mask as a .npy file
            np.save(mask_path, mask)

        # Determine windows for this image    
        windows = make_windows(W, H, patch_size=512, stride=512)

        # Split into train/validation
        split = int(0.8 * len(windows))
        train_wins = windows[:split]
        val_wins   = windows[split:]

        # Create generators
        train_seq = PatchSequence(img_path, mask_path, train_wins, batch_size=4)
        val_seq   = PatchSequence(img_path, mask_path, val_wins,   batch_size=4)

        # Checkpoint per image
        chkpt = ModelCheckpoint(
            f'unet_best_{os.path.splitext(fname)[0]}.h5',
            monitor='val_accuracy', save_best_only=True, verbose=1
        )

        # Train for a few epochs on this image
        print(f"Training on {fname}: {len(train_seq)} steps, {len(val_seq)} val steps")
        model.fit(
            train_seq,
            epochs=5,
            steps_per_epoch=len(train_seq),
            validation_data=val_seq,
            validation_steps=len(val_seq),
            callbacks=[chkpt]
        )
        # Log the processed image path
        with open('processed_images.log', 'a') as log_file:
            log_file.write(f"{img_path}\n")

        # Save stats and timestamp to a log file
        with open('time.log', 'a') as time_log:
            time_log.write(f"Processed {img_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            # Write some stats
            time_log.write(f"Image shape: {H}x{W}, Windows: {len(windows)}, Train steps: {len(train_seq)}, Val steps: {len(val_seq)}\n")
            # Save info about accuracy and loss
            train_metrics = model.evaluate(train_seq, steps=len(train_seq), verbose=0)
            val_metrics = model.evaluate(val_seq, steps=len(val_seq), verbose=0)
            time_log.write(f"Train accuracy: {train_metrics[1]}, Val accuracy: {val_metrics[1]}\n")

        # Save cumulative weights to resume on next images
        model.save_weights('unet_attention_weights.weights.h5')

    
if __name__ == '__main__':
    # Directories containing all large GeoTIFFs and their .npy mask arrays
    img_dir = '/home/epele/.cache/kagglehub/datasets/isaienkov/deforestation-in-ukraine/versions/1'

    df = gpd.read_file(img_dir + "/deforestation_labels.geojson")
    # Instantiate model once, optionally load existing weights to resume
    model = UNetAM(input_size=(512,512,3), drop_rate=0.25, lr=5e-4, filter_base=16)
    # model.load_weights('unet_attention_weights.h5')  # if resuming

    # Log the starting time
    with open('time.log', 'a') as time_log:
        time_log.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # Load processed image paths from log file if it exists
    processed_images = []
    if os.path.exists('processed_images.log'):
        with open('processed_images.log', 'r') as log_file:
            processed_images = [line.strip() for line in log_file.readlines()]

    training_folders = []

    # Loop over each image/mask pair and train
    for folder in sorted(os.listdir(img_dir)):

        # Skip if folder is not a directory
        if not os.path.isdir(os.path.join(img_dir, folder)):
            print(f"Skipping {folder} as it is not a directory.")
            continue
        safe_folders = [f for f in os.listdir(os.path.join(img_dir, folder)) if f.endswith('.SAFE')]
        if len(safe_folders) != 1:
            print(f"Skipping {folder} as it does not contain exactly one .SAFE folder.")
            continue
        safe_folder = safe_folders[0]
        if not os.path.exists(os.path.join(img_dir, folder, safe_folder)):
            print(f"SAFE folder not found in {folder}")
            continue
        granule_dir = os.path.join(img_dir, folder, safe_folder, 'GRANULE')
        if not os.path.exists(granule_dir):
            print(f"GRANULE folder not found in {folder}")
            continue
        for granule in os.listdir(granule_dir):
            img_data_dir = os.path.join(granule_dir, granule, 'IMG_DATA')
            if not os.path.exists(img_data_dir):
                print(f"IMG_DATA folder not found in {granule}")
                continue
            training_folders.append(img_data_dir)

    for training_folder in training_folders:
        print(f"Training on folder: {training_folder}")
        train_per_folder(model, training_folder, df, processed_images)

    # Finally save the fully trained model
    model.save('unet_attention_ukr_all.h5')
