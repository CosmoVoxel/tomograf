import streamlit as st
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime
import io
import time

from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
import pydicom.uid

st.set_page_config(page_title="Tomograph Simulator", layout="wide")
st.title("Tomograph Simulator")

if "image" not in st.session_state:
    st.session_state.image = None
if "image_name" not in st.session_state:
    st.session_state.image_name = None
if "sinogram" not in st.session_state:
    st.session_state.sinogram = None
if "reconstructed" not in st.session_state:
    st.session_state.reconstructed = None
if "patient_name" not in st.session_state:
    st.session_state.patient_name = "Wlad"
if "patient_id" not in st.session_state:
    st.session_state.patient_id = "3821"
if "study_date" not in st.session_state:
    st.session_state.study_date = datetime.date.today()
if "comments" not in st.session_state:
    st.session_state.comments = "Simulation result"
if "animating" not in st.session_state:
    st.session_state.animating = False


def load_image(uploaded_file):
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            ds = pydicom.dcmread(uploaded_file, force=True)
            image_array = ds.pixel_array

            if image_array.dtype != np.uint8:
                image_array = image_array.astype(float)
                min_val = np.min(image_array)
                max_val = np.max(image_array)
                if max_val > min_val:
                    image_array = (image_array - min_val) / (max_val - min_val) * 255.0
                else:
                    image_array = np.zeros_like(image_array)
                image_array = image_array.astype(np.uint8)

            try:
                st.session_state.patient_name = str(ds.PatientName)
            except:
                pass
            try:
                st.session_state.patient_id = str(ds.PatientID)
            except:
                pass
            try:
                st.session_state.study_date = datetime.datetime.strptime(
                    ds.StudyDate, "%Y%m%d"
                ).date()
            except:
                pass
            try:
                st.session_state.comments = str(
                    ds.StudyDescription or ds.ImageComments or "DICOM"
                )
            except:
                pass

        except Exception as e:
            try:
                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
                image_array = np.array(image.convert("L"))

                st.session_state.patient_name = "None"
                st.session_state.patient_id = "12345"
                st.session_state.study_date = datetime.date.today()
                st.session_state.comments = "Sim Res"
            except Exception as e_img:
                st.error(f"Error loading image: {e_img}")
                return None
        return image_array
    return None


def normalize_for_display(image_array):
    if image_array is None or image_array.size == 0:
        return None

    try:
        img_float = image_array.astype(float)
        normalized = rescale_intensity(img_float, out_range=(0, 255))
        return normalized.astype(np.uint8)
    except Exception as e:
        st.error(f"Error normalizing image for display: {e}")

        return (
            np.zeros_like(image_array).astype(np.uint8)
            if image_array is not None
            else None
        )


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points

# Parallel projection (multiple emissions)
def create_sinogram(
    image,
    num_projections,
    num_detectors,
    detector_spacing,
    max_projection_index=None,
    single_projection_index=None,
):
    height, width = image.shape
    center_x, center_y = width // 2, height // 2
    max_radius = np.sqrt(height**2 + width**2) / 2

    if single_projection_index is not None:
        start_index = single_projection_index
        end_index = single_projection_index + 1

        sinogram = (
            np.zeros((1, num_detectors))
            if single_projection_index is not None
            else np.zeros((num_projections, num_detectors))
        )
    else:
        start_index = 0
        end_index = (
            max_projection_index + 1
            if max_projection_index is not None
            else num_projections
        )
        sinogram = np.zeros((num_projections, num_detectors))

    angles = np.linspace(0, 180, num_projections, endpoint=False)

    for i in range(start_index, end_index):
        angle_deg = angles[i]
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        detector_line_direction = (-sin_a, cos_a)

        for j in range(num_detectors):
            detector_offset = (j - (num_detectors - 1) / 2) * detector_spacing
            detector_pos_x = center_x + detector_offset * detector_line_direction[0]
            detector_pos_y = center_y + detector_offset * detector_line_direction[1]
            x_start = detector_pos_x - max_radius * 1.1 * cos_a
            y_start = detector_pos_y - max_radius * 1.1 * sin_a
            x_end = detector_pos_x + max_radius * 1.1 * cos_a
            y_end = detector_pos_y + max_radius * 1.1 * sin_a
            points = bresenham_line(
                int(round(x_start)),
                int(round(y_start)),
                int(round(x_end)),
                int(round(y_end)),
            )

            line_integral = 0.0
            for x, y in points:
                if 0 <= y < height and 0 <= x < width:
                    line_integral += image[y, x]

            row_index = 0 if single_projection_index is not None else i
            sinogram[row_index, j] = line_integral

    if (
        single_projection_index is None
        and max_projection_index is not None
        and end_index < num_projections
    ):
        sinogram[end_index:, :] = 0

    return sinogram

"""
Standard Backprojection (Nearest Neighbor) Implementation.
For each angle:
    - Get the corresponding projection line from the sinogram.
    - For each pixel in the output image:
        - Calculate its projection onto the detector line.
        - Find the nearest detector index.
        - Add the value from the sinogram at that detector index to the pixel.
"""
def backproject(
    sinogram,
    num_projections,
    num_detectors,
    detector_spacing,
    output_shape,
    max_projection_index=None,
    single_projection_index=None,
):
    height, width = output_shape
    center_x, center_y = width // 2, height // 2
    reconstructed = np.zeros(output_shape, dtype=float)

    angles = np.linspace(0, 180, num_projections, endpoint=False)

    # Determine the range of projection indices to process
    if single_projection_index is not None:
        start_index = single_projection_index
        end_index = single_projection_index + 1
    else:
        start_index = 0
        end_index = (
            max_projection_index + 1
            if max_projection_index is not None
            else num_projections
        )

    y_coords, x_coords = np.indices(output_shape)
    x_centered = x_coords - center_x
    y_centered = y_coords - center_y

    for i in range(start_index, end_index):
        angle_rad = np.radians(angles[i])
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        pixel_projection_offset = x_centered * (-sin_a) + y_centered * cos_a
        
        detector_indices_float = (
            pixel_projection_offset / detector_spacing + (num_detectors - 1) / 2
        )

        # (Nearest Neighbor)
        j_nearest = np.round(detector_indices_float).astype(int)

        valid_mask = (j_nearest >= 0) & (j_nearest < num_detectors)

        # Initialize the contribution from this projection angle
        projection_contribution = np.zeros(output_shape, dtype=float)

        # Get the sinogram values corresponding to the nearest valid detector indices
        # np.take efficiently gathers elements using the indices
        sinogram_values = np.take(sinogram[i, :], j_nearest[valid_mask])

        # Add these sinogram values to the corresponding pixels in the projection contribution map
        projection_contribution[valid_mask] = sinogram_values

        # Add (backproject) the contribution of this angle to the reconstructed image
        reconstructed += projection_contribution

    return reconstructed


def convert_image_to_ubyte(img):
    if img.dtype == np.uint8:
        return img

    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))


def save_dicom(img, patient_data):
    if img is None:
        st.error("No image data to save.")
        return None

    try:

        img_converted = convert_image_to_ubyte(img)

        meta = Dataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        meta.ImplementationClassUID = pydicom.uid.generate_uid()

        ds = FileDataset(None, {}, preamble=b"\0" * 128)
        ds.file_meta = meta

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID

        ds.PatientName = patient_data.get("PatientName", "None")
        ds.PatientID = patient_data.get("PatientID", "NoID")
        ds.ImageComments = patient_data.get("ImageComments", "-")

        ds.Modality = "CT"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

        ds.Rows, ds.Columns = img_converted.shape
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7

        now = datetime.datetime.now()
        ds.StudyDate = patient_data.get("StudyDate", now.strftime("%Y%m%d"))
        ds.StudyTime = now.strftime("%H%M%S.%f")
        ds.InstanceNumber = 1
        ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

        ds.PixelData = img_converted.tobytes()

        buffer = io.BytesIO()
        ds.save_as(buffer, write_like_original=False)
        buffer.seek(0)
        return buffer

    except Exception as e:
        st.error(f"Error creating DICOM: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_calculation(
    calc_func,
    calc_args,
    result_key,
    placeholder,
    title,
    show_steps,
    num_steps,
    progress_label="Projection",
):
    if show_steps:
        st.session_state.animating = True
        status_msg = f"Animating {title.lower()} creation..."
        calculation_error = False
        start_time = time.time()

        if result_key == "sinogram":
            temp_result = np.zeros((num_steps, calc_args["num_detectors"]))
        else:
            temp_result = np.zeros(calc_args["output_shape"], dtype=float)

        with placeholder.container():
            st.subheader(title)
            img_placeholder = st.empty()
            caption_placeholder = st.empty()
            progress_placeholder = st.empty()

        with st.spinner(status_msg):
            last_update_time = time.time()
            for i in range(num_steps):
                step_args = calc_args.copy()
                step_args["single_projection_index"] = i
                try:
                    step_result = calc_func(**step_args)
                    if step_result is None or step_result.size == 0:
                        raise ValueError("Calculation step returned empty result")

                    if result_key == "sinogram":
                        temp_result[i, :] = step_result[0, :]
                    else:
                        temp_result += step_result

                except Exception as e:
                    st.error(f"Error during {title.lower()} step {i+1}: {e}")
                    calculation_error = True
                    break

                current_time = time.time()
                if current_time - last_update_time > 0.1 or i == num_steps - 1:
                    with placeholder.container():
                        st.subheader(title)
                        img_placeholder.image(
                            normalize_for_display(temp_result.copy()),
                            use_container_width=True,
                        )
                        caption_placeholder.caption(
                            f"Size: {temp_result.shape} | {progress_label}: {i+1}/{num_steps}"
                        )
                        progress_placeholder.progress((i + 1) / num_steps)
                    last_update_time = current_time

            st.session_state[result_key] = temp_result
            end_time = time.time()
            with placeholder.container():
                st.subheader(title)
                img_placeholder.image(
                    normalize_for_display(st.session_state[result_key]),
                    use_container_width=True,
                )
                caption_placeholder.caption(
                    f"Size: {st.session_state[result_key].shape}"
                )
                progress_placeholder.empty()
                if calculation_error:
                    st.error(f"{title} creation interrupted due to error.")
                else:
                    st.success(
                        f"{title} animation completed in {end_time - start_time:.2f} sec."
                    )

        return True

    else:
        status_msg = (
            f"Calculating {title.lower()}... ({num_steps} {progress_label.lower()}s)"
        )
        with placeholder.container():
            st.subheader(title)
            img_placeholder = st.empty()
            caption_placeholder = st.empty()
            img_placeholder.info("Calculating...")
            with st.spinner(status_msg):
                start_time = time.time()
                full_args = calc_args.copy()
                full_args.pop("single_projection_index", None)
                try:
                    result = calc_func(**full_args)
                    st.session_state[result_key] = result
                    end_time = time.time()
                    success = result is not None
                except Exception as e:
                    st.error(f"Error calculating {title.lower()}: {e}")
                    st.session_state[result_key] = None
                    success = False
                    end_time = time.time()

            if success:
                img_placeholder.image(
                    normalize_for_display(st.session_state[result_key]),
                    use_container_width=True,
                )
                caption_placeholder.caption(
                    f"Size: {st.session_state[result_key].shape}"
                )
                st.success(f"{title} calculated in {end_time - start_time:.2f} sec.")
            else:
                img_placeholder.error(f"Failed to calculate {title}.")
        return success


# GUI STUFF
st.sidebar.header("Simulation Parameters")

uploaded_file = st.sidebar.file_uploader(
    "Upload an image (JPG, PNG, DCM)", type=["jpg", "jpeg", "png", "dcm"]
)

current_file_key = (
    f"{uploaded_file.file_id}-{uploaded_file.name}-{uploaded_file.size}"
    if uploaded_file
    else None
)
if current_file_key != st.session_state.get("image_key"):
    st.session_state.image = load_image(uploaded_file)
    st.session_state.image_key = current_file_key

    st.session_state.sinogram = None
    st.session_state.reconstructed = None

    st.session_state.pop("create_sinogram_step", None)
    st.session_state.pop("recon_step", None)

    if uploaded_file:
        st.rerun()

angular_step = st.sidebar.slider(
    "Angular Step Δα (°)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    key="angular_step",
)
num_projections = max(1, int(180.0 / angular_step))
st.sidebar.caption(f"Number of projections: {num_projections} (for 180°)")

num_detectors = st.sidebar.slider(
    "Number of detectors (per projection)",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    key="num_detectors",
)
detector_spacing = st.sidebar.slider(
    "Detector spacing (pixels)",
    min_value=0.1,
    max_value=20.0,
    value=3.0,
    step=0.1,
    key="detector_spacing",
)
st.sidebar.caption(f"Total coverage: {(num_detectors-1)*detector_spacing:.1f} pixels")

show_steps = st.sidebar.checkbox("Show intermediate steps", key="show_steps")

st.sidebar.header("DICOM Information")
st.session_state.patient_name = st.sidebar.text_input(
    "Patient Name", value=st.session_state.patient_name, key="dicom_patient_name"
)
st.session_state.patient_id = st.sidebar.text_input(
    "Patient ID", value=st.session_state.patient_id, key="dicom_patient_id"
)
st.session_state.study_date = st.sidebar.date_input(
    "Study Date", value=st.session_state.study_date, key="dicom_study_date"
)
st.session_state.comments = st.sidebar.text_area(
    "Comments", value=st.session_state.comments, key="dicom_comments"
)

if st.session_state.image is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(
            normalize_for_display(st.session_state.image),
            caption=f"Size: {st.session_state.image.shape}",
            use_container_width=True,
        )

    control_container = st.container()
    with control_container:
        c1, c2, c3 = st.columns([1, 1, 1])
        is_animating = st.session_state.get("animating", False)
        sinogram_button_pressed = c1.button(
            "Create Sinogram", key="run_sinogram_btn", disabled=is_animating
        )
        recon_button_pressed = c2.button(
            "Reconstruct Image",
            key="run_recon_btn",
            disabled=st.session_state.get("sinogram") is None or is_animating,
        )
        c3.empty()

    rerun_needed = False
    placeholder_col2 = col2.empty()
    placeholder_col3 = col3.empty()

    try:
        if sinogram_button_pressed:
            st.session_state.reconstructed = None
            st.session_state.sinogram = None
            sinogram_args = {
                "image": st.session_state.image,
                "num_projections": num_projections,
                "num_detectors": num_detectors,
                "detector_spacing": detector_spacing,
                "max_projection_index": None,
            }
            success = run_calculation(
                calc_func=create_sinogram,
                calc_args=sinogram_args,
                result_key="sinogram",
                placeholder=placeholder_col2,
                title="Sinogram",
                show_steps=show_steps,
                num_steps=num_projections,
                progress_label="Projection",
            )
            if success:
                rerun_needed = True

        if recon_button_pressed:
            if st.session_state.get("sinogram") is None:
                st.warning("Sinogram is missing. Please create sinogram first.")
            else:
                st.session_state.reconstructed = None
                recon_args = {
                    "sinogram": st.session_state.sinogram,
                    "num_projections": num_projections,
                    "num_detectors": num_detectors,
                    "detector_spacing": detector_spacing,
                    "output_shape": st.session_state.image.shape,
                    "max_projection_index": None,
                }
                success = run_calculation(
                    calc_func=backproject,
                    calc_args=recon_args,
                    result_key="reconstructed",
                    placeholder=placeholder_col3,
                    title="Reconstructed Image",
                    show_steps=show_steps,
                    num_steps=num_projections,
                    progress_label="Projection",
                )
                if success:
                    rerun_needed = True

    finally:

        if st.session_state.get("animating", False):
            st.session_state.animating = False

            if not rerun_needed:
                print("Rerun needed post-finally to update button states.")
                rerun_needed = True

    with col2:
        st.subheader("Sinogram")
        if st.session_state.get("sinogram") is not None:
            st.image(
                normalize_for_display(st.session_state.sinogram),
                caption=f"Size: {st.session_state.sinogram.shape}",
                use_container_width=True,
            )

        else:

            st.empty()

    with col3:
        st.subheader("Reconstructed Image")
        if st.session_state.get("reconstructed") is not None:
            st.image(
                normalize_for_display(st.session_state.reconstructed),
                caption=f"Size: {st.session_state.reconstructed.shape}",
                use_container_width=True,
            )

            try:

                patient_data = {
                    "PatientName": st.session_state.patient_name,
                    "PatientID": st.session_state.patient_id,
                    "ImageComments": st.session_state.comments,
                    "StudyDate": st.session_state.study_date.strftime("%Y%m%d"),
                }

                dicom_buffer = save_dicom(
                    img=st.session_state.reconstructed, patient_data=patient_data
                )

                if dicom_buffer:
                    st.download_button(
                        label="Save as DICOM",
                        data=dicom_buffer,
                        file_name=f"reconstructed_{st.session_state.patient_id}_{st.session_state.study_date}.dcm",
                        mime="application/dicom",
                        key="download_dicom_btn",
                    )
            except Exception as e:
                st.error(f"Error preparing DICOM file: {e}")

        else:

            st.empty()

    if rerun_needed:
        print(f"Rerun triggered due to calculation/animation completion.")
        st.rerun()

else:
    st.info("Please upload an image to start the simulation.")
