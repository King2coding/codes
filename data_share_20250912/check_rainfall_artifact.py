#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import requests
import xarray as xr


TOKEN = ""
DOWNLOAD_URL = "https://cml.tahmo.org/api/v1/rainfall-maps/dd764a3e-db3a-4305-8362-e2ed3cca668d/download/"
OUTPUT_PATH = Path("rainfall_artifact.nc")
IMAGE_PATH = Path("rainfall_artifact_preview.png")


def download_file():
    headers = {
        "Authorization": f"Bearer {TOKEN}",
    }

    with requests.get(DOWNLOAD_URL, headers=headers, stream=True, timeout=300) as response:
        response.raise_for_status()
        with OUTPUT_PATH.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)

    print(f"Downloaded: {OUTPUT_PATH.resolve()}")
    print(f"Size: {OUTPUT_PATH.stat().st_size} bytes")


def inspect_netcdf():
    with xr.open_dataset(OUTPUT_PATH) as dataset:
        print("\nNetCDF opened successfully.")
        print(f"Variables: {list(dataset.data_vars)}")
        print(f"Coordinates: {list(dataset.coords)}")
        print(f"Dimensions: {dict(dataset.sizes)}")

        if dataset.attrs:
            print("\nGlobal attributes:")
            for key, value in dataset.attrs.items():
                print(f"  {key}: {value}")

        print("\nDataset summary:")
        print(dataset)


def pick_rainfall_variable(dataset):
    preferred_names = [
        "rainfall",
        "rain_rate",
        "R",
        "R_da",
        "R_mm_per_h",
        "rainfall_rate",
    ]

    for name in preferred_names:
        if name in dataset.data_vars:
            return name

    for name in dataset.data_vars:
        lowered = name.lower()
        if "rain" in lowered or lowered.startswith("r"):
            return name

    return next(iter(dataset.data_vars))


def generate_image():
    with xr.open_dataset(OUTPUT_PATH) as dataset:
        if not dataset.data_vars:
            raise ValueError("No data variables were found in the NetCDF file.")

        variable_name = pick_rainfall_variable(dataset)
        data_array = dataset[variable_name]

        if "time" in data_array.dims:
            selected_time = data_array["time"].isel(time=0).item()
            data_array = data_array.isel(time=0)
        else:
            selected_time = None

        data_array = data_array.squeeze(drop=True)

        if data_array.ndim != 2:
            raise ValueError(
                f"Cannot generate a 2D image from variable '{variable_name}' with shape {data_array.shape}."
            )

        figure, axis = plt.subplots(figsize=(10, 6))
        data_array.plot(ax=axis, cmap="Spectral_r", vmax=15, add_colorbar=True)
        title = f"{variable_name} rainfall map"
        if selected_time is not None:
            title = f"{title} | {selected_time}"
        axis.set_title(title)
        axis.set_xlabel("lon" if "lon" in data_array.coords else data_array.dims[-1])
        axis.set_ylabel("lat" if "lat" in data_array.coords else data_array.dims[0])
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(IMAGE_PATH, dpi=200, bbox_inches="tight")
        plt.close(figure)

    print(f"\nImage saved: {IMAGE_PATH.resolve()}")


def main():
    if "PASTE_BEARER_TOKEN_HERE" in TOKEN or "YOUR_DOMAIN" in DOWNLOAD_URL:
        raise ValueError("Update TOKEN and DOWNLOAD_URL at the top of the script before running it.")

    download_file()
    inspect_netcdf()
    generate_image()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
