# Tomograph Simulator

Hey there! :]

This little app is a basic simulator for how a CT (Computed Tomography) scanner works. You feed it an image (like a photo or even a medical DICOM file), and it pretends to scan it.

It creates **sinogram** (which is like the raw data the scanner gets) and then tries to **reconstruct** the original image back from that sinogram.

**IMPORTANT NOTE!**
- No filters are applied after Inverse Radon Transformation

## Getting it Running
0. **Git Clone**: Clone Repo! ;}
    ```bash
    git clone https://github.com/CosmoVoxel/tomograf.git
    ```

1.  **Install the Goods**: Open your terminal or command prompt and run this:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App**: Navigate to the folder where you saved `app.py` in your terminal and run:
    ```bash
    streamlit run app.py
    ```
    Your web browser should pop open with the app!


### Links:
- https://en.wikipedia.org/wiki/Tomographic_reconstruction
- https://www.mathworks.com/help/images/radon-transform.html
- https://www.mathworks.com/help/images/the-inverse-radon-transformation.html  

## How to Use It


*   **Upload Image**: Use the sidebar on the left to upload an image file. It can handle common types like JPG, PNG, and also DICOM (.dcm) files. If you upload a DICOM, it'll try to grab patient info from it.
*   **Tweak Parameters**:
    *   `Angular Step`: How much the pretend scanner rotates between each 'shot'. Smaller angles mean more projections (usually better quality, but slower).
    *   `Number of detectors`: How many sensors collect data for each projection.
    *   `Detector spacing`: How far apart the sensors are.
*   **See the steps?**: Check the `Show intermediate steps` box if you want to watch the sinogram and reconstruction build up step-by-step. It's neat but takes longer.
*   **Patient Info**: Fill this stuff in if you want it saved in the final DICOM file you download later.
*   **Buttons**:
    1.  Click `Create Sinogram` first. This generates the projection data from your image based on the parameters you set.
    2.  Once the sinogram is ready, click `Reconstruct Image`. This uses the sinogram data to build the final image.
*   **Download**: If the reconstruction looks okay you'll see a `Save as DICOM` button appear below it. Click that to save your result.

## GUI - *streamlit*


## Cool Stuff It Does

*   Loads JPG, PNG, and **DICOM** images.
*   Tries to read patient info from DICOMs.
*   Shows animated step-by-step processing (optional).
*   Saves the final reconstructed image as a DICOM file.

## What You Need (Requirements)

*   Python (Version 3.x recommended)