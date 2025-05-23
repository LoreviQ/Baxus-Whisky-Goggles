import pytest
import pandas as pd
from dataset.images import fetch_images

@pytest.fixture
def mock_data_folder(tmp_path):
    """Fixture to create a temporary mock data folder."""
    data_folder = tmp_path / "data"
    data_folder.mkdir()

    # Create a mock dataset.tsv file
    dataset_path = data_folder / "dataset.tsv"
    dataset_path.write_text(
        "id\tname\tsize\tproof\tadv\tspirit_type\tbrand_id\tpopularity\tavg_msrp\tfair_price\tshelf_price\ttotal_score\twishlist_count\tvote_count\tbar_count\tranking\timage_url\n" +
        "1\tWhiskey A\t750\t40.0\t0.0\tWhiskey\t101\t100\t50.0\t45.0\t55.0\t90\t10\t5\t2\t1\thttp://example.com/image1.png\n" +
        "2\tWhiskey B\t500\t35.0\t0.0\tBourbon\t102\t200\t60.0\t50.0\t70.0\t85\t20\t10\t5\t2\thttp://example.com/image2.png\n"
    )

    # Create a mock source_images directory
    image_dir = data_folder / "source_images"
    image_dir.mkdir()

    return data_folder

def test_fetch_images(mock_data_folder, requests_mock):
    """Test the fetch_images function."""
    # Mock image URLs
    requests_mock.get("http://example.com/image1.png", content=b"fake_image_data_1")
    requests_mock.get("http://example.com/image2.png", content=b"fake_image_data_2")

    # Load the dataset
    dataset_path = mock_data_folder / "dataset.tsv"
    dataframe = pd.read_csv(dataset_path, sep='\t')

    # Call fetch_images
    image_dir = mock_data_folder / "source_images"
    fetch_images(str(image_dir), dataframe)

    # Check if images are saved correctly
    image1_path = image_dir / "1.png"
    image2_path = image_dir / "2.png"

    assert image1_path.exists()
    assert image2_path.exists()
    assert image1_path.read_bytes() == b"fake_image_data_1"
    assert image2_path.read_bytes() == b"fake_image_data_2"