from pydantic import BaseModel


class ImageGenerationRequest(BaseModel):
    prompt: str


class ImageSimilarityRequest(BaseModel):
    prompt: str
    image_url: str


class TextToSpeechRequest(BaseModel):
    text: str
