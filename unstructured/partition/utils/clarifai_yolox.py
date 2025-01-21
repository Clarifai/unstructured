from clarifai.client.model import Model
from PIL import Image as PILImage
from unstructured_inference.constants import Source
from unstructured_inference.inference.layoutelement import LayoutElement, LayoutElements
from unstructured_inference.models.unstructuredmodel import UnstructuredObjectDetectionModel

from unstructured.partition.utils.constants import LAYOUT_DEFAULT_CLARIFAI_MODEL_URL


class ClarifaiYoloXModel(UnstructuredObjectDetectionModel):
    """Clarifai YoloX model for layout segmentation."""

    def __init__(self):
        self.model = Model(LAYOUT_DEFAULT_CLARIFAI_MODEL_URL)
        self.confidence_threshold = 0.1

    def predict(self, x: PILImage.Image) -> LayoutElements:
        """Predict using Clarifai YoloX model."""
        image_bytes = self.pil_image_to_bytes(x)
        model_prediction = self.model.predict_by_bytes(image_bytes, input_type="image")
        return self.parse_data(model_prediction, x)

    def initialize(self):
        pass

    def parse_data(
        self,
        model_prediction,
        image: PILImage.Image,
    ) -> LayoutElements:
        """Process model prediction output into Unstructured class. Bounding box coordinates
        are converted to original image resolution. Layouts are filtered based on confidence
        threshold.
        """
        regions_data = model_prediction.outputs[0].data.regions
        regions = []
        input_w, input_h = image.size
        for region in regions_data:
            bboxes = region.region_info.bounding_box
            y1, x1, y2, x2 = bboxes.top_row, bboxes.left_col, bboxes.bottom_row, bboxes.right_col
            detected_class = region.data.concepts[0].name
            confidence = region.value
            if confidence >= self.confidence_threshold:
                region = LayoutElement.from_coords(
                    x1 * input_w,
                    y1 * input_h,
                    x2 * input_w,
                    y2 * input_h,
                    text=None,
                    type=detected_class,
                    prob=confidence,
                    source=Source.YOLOX,
                )
                regions.append(region)

        regions.sort(key=lambda element: element.bbox.y1)
        return LayoutElements.from_list(regions)

    def pil_image_to_bytes(self, image: PILImage) -> bytes:
        from io import BytesIO

        with BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()
