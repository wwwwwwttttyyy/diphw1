import os
import subprocess
import argparse
import shutil
import sys
from pathlib import Path

# Allow COLMAP (Qt-based) to run on headless servers without an X display.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def run_colmap_cli(data_dir):
    """Run the original COLMAP command-line pipeline."""
    database_path = os.path.join(data_dir, 'database.db')
    image_path = os.path.join(data_dir, 'images')
    sparse_path = os.path.join(data_dir, 'sparse')

    subprocess.run(['colmap', 'feature_extractor', '--image_path', image_path,
                    '--database_path', database_path,
                    '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_model', 'PINHOLE',
                    '--SiftExtraction.use_gpu', '0'], check=True)
    subprocess.run(['colmap', 'exhaustive_matcher', '--database_path', database_path,
                    '--SiftMatching.use_gpu', '0'], check=True)
    os.makedirs(sparse_path, exist_ok=True)
    subprocess.run(['colmap', 'mapper', '--image_path', image_path,
                    '--database_path', database_path,
                    '--output_path', sparse_path], check=True)

    text_path = os.path.join(sparse_path, '0_text')
    os.makedirs(text_path, exist_ok=True)
    subprocess.run(['colmap', 'model_converter',
                    '--input_path', os.path.join(sparse_path, '0'),
                    '--output_path', text_path, '--output_type', 'TXT'], check=True)


def run_pycolmap(data_dir):
    """Run the same sparse reconstruction through local PyCOLMAP bindings."""
    vendor_path = Path(__file__).resolve().parent / '.vendor'
    if vendor_path.is_dir():
        sys.path.insert(0, str(vendor_path))

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError(
            'Neither the colmap command nor PyCOLMAP is available.'
        ) from exc

    database_path = Path(data_dir) / 'database.db'
    image_path = Path(data_dir) / 'images'
    sparse_path = Path(data_dir) / 'sparse'
    sparse_path.mkdir(parents=True, exist_ok=True)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = 'PINHOLE'
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        device=pycolmap.Device.cpu,
    )
    pycolmap.match_exhaustive(
        database_path=database_path,
        device=pycolmap.Device.cpu,
    )
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=sparse_path,
    )
    if not reconstructions:
        raise RuntimeError('COLMAP could not recover a sparse reconstruction.')

    _, reconstruction = max(
        reconstructions.items(), key=lambda item: item[1].num_reg_images()
    )
    text_path = sparse_path / '0_text'
    text_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(text_path)
    print(
        f'Reconstructed {reconstruction.num_reg_images()} images and '
        f'{reconstruction.num_points3D()} sparse points.'
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run COLMAP for multi-view stereo')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input directory containing images in data_dir/images')
    args = parser.parse_args()
    data_dir = args.data_dir

    if shutil.which('colmap'):
        run_colmap_cli(data_dir)
    else:
        print('COLMAP command not found; using local PyCOLMAP bindings.')
        run_pycolmap(data_dir)

    print("COLMAP multi-view stereo pipeline completed successfully!")
    print("Sparse 3D reconstruction saved in:", os.path.join(data_dir, 'sparse', '0_text'))
    
