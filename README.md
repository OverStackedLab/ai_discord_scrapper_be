# ASR Automatic Speech Recognition

- Running the project

```
$ uv run modal serve src.modal_app.main
```

# Understanding Multi-Modal AI Systems: Architecture and Implementation

Multi-modal AI systems combine different types of data processing - text, images, audio, and more - into unified applications. Let's explore how these systems work and how to build them effectively.

## Core Concepts of Multi-Modal AI

Multi-modal AI refers to systems that can process and generate multiple types of media. Think of it like having different specialized processors working together: one for understanding speech, another for analyzing images, another for text, and so on. These components often communicate through a common interface—typically text—and orchestrate tasks such as generating text from images (image captions), describing images with speech, or answering questions about images. The growing popularity of multi-modal models has also sparked significant interest in the open-source community—projects like [Mistral](https://mistral.ai/) (for text generation) and [Moondream1](https://x.com/vikhyatk/status/1749625143155167702?s=20) (for vision-language) illustrate some of the latest frontiers.

### The Architecture of Multi-Modal Systems

Modern multi-modal systems typically follow a pipeline architecture:

1. **Input Processing**: Converting various inputs (speech, images, text) into a format the system can process.
2. **Core Processing**: Using specialized models to understand or generate content.
   - Several open-source expansions have emerged, such as [Griptape](https://github.com/griptape-ai/griptape) and [Crew AI](https://github.com/joaomdmoura/crewAI), which help orchestrate multiple AI agents in tandem. Not relevant for our particular use case as we get into agents in a future module, but worth noting that multi-modal setups are great candidates for agents.
3. **Output Generation**: Converting the processed results back into the desired format.
   - This includes everything from speech synthesis (Text-to-Speech) to image generation (Text-to-Image) and more.

A noteworthy development has been the rise of Mixture-of-Experts (MoE) architectures (like [DeepSeek MOE](https://twitter.com/deepseek_ai/status/1745304852211839163), [Mamba MOE](https://arxiv.org/abs/2401.04081), and others), which allow for more specialized “experts” within a large model, improving performance on different modalities and tasks.

Let's look at each component in detail.

## Component Breakdown and Implementation

For each code snippet, begin again with a new branch starting from the `main` template that we started with in the beginning of the course. For each snippet, we’ll highlight relevant open-source tools and notes that have arisen over the past year, as well as alternative frameworks.

> **Note:** If you’re curious about the latest generation of models and open-source options for these tasks, check out resources like [Hugging Face](https://huggingface.co/models?sort=trending)

### Installing Dependencies

Run this to install everything that we'll need for this project!

```bash
uv add sentence-transformers pydantic Pillow openai python-multipart
```

in your `modal_app` directory, alongside `main.py` and `common.py` create a `models.py` file and fill it with the following

```python
# src/modal_app/models.py
from pydantic import BaseModel

class ImageGenerationRequest(BaseModel):
    prompt: str

class ImageSimilarityRequest(BaseModel):
    prompt: str
    image_url: str

class TextToSpeechRequest(BaseModel):
    text: str
```

The code creates three model classes that specify the expected structure of data:

- `ImageGenerationRequest`: Expects a text prompt
- `ImageSimilarityRequest`: Expects a prompt and an image URL
- `TextToSpeechRequest`: Expects text input

When you use these models in your routes, Pydantic automatically:

- Validates that incoming requests have the required fields
- Ensures fields are the correct type (all strings in this case)
- Converts JSON data into Python objects

It's like having a bouncer that checks if data matches your requirements before it enters your application. This helps catch errors early and makes your code more reliable.
We will use these pydantic models to give us a bit of type safety within our routes.
While not strictly necessary, it's certainly best practice!

at the top of `main.py` place these imports. You can replace the previous imports that you began with.

```python
# src/modal_app/main.py

import sqlite3
from io import BytesIO
from urllib.request import urlopen
import base64
import os

from modal import asgi_app
from PIL import Image
from fastapi import UploadFile, File, HTTPException
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from .common import DB_PATH, VOLUME_DIR, app, fastapi_app, volume
from .models import ImageGenerationRequest, ImageSimilarityRequest, TextToSpeechRequest
```

### Speech Recognition (Speech → Text)

Speech recognition converts audio input into text. The industry standard is OpenAI's Whisper, available through their API or as an open-source model. Other notable fine-tunings/forks have gained popularity though such as:

- **whisper.cpp** for on-device usage ([GitHub](https://github.com/ggerganov/whisper.cpp))
  - [use it in the browser](https://whisper.ggerganov.com/)!
- **Faster-Whisper** for better speed ([GitHub](https://github.com/guillaumekln/faster-whisper))
- **whisper-large-v3-turbo** for state-of-the-art improvements ([Hugging Face](https://huggingface.co/openai/whisper-large-v3-turbo))
- [Trending ASR models on Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)

```python
# src/modal_app/main.py
@fastapi_app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file. Received: " + file.content_type
            )
    try:
        audio_bytes = await file.read()
        audio_file = BytesIO(audio_bytes)
        audio_file.name = file.filename or "audio.webm"

        # Print some debug info
        print(f"Processing audio file: {file.filename}")
        print(f"Content type: {file.content_type}")
        print(f"File size: {len(audio_bytes)} bytes")

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return {"transcript": transcription.text}
    except Exception as e:
        print("there was an error")
        print(str(e))
        return {"error": str(e)}, 500
```

> **Tip:** If you prefer open-source speech recognition to avoid sending data to a closed API, try hooking up [whisper.cpp](https://github.com/ggerganov/whisper.cpp) or [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) on your local container. A good example is [JohnTheNerd’s local LLM voice assistant project](https://johnthenerd.com/blog/local-llm-assistant/) which also uses on-device speech solutions.

### Image Generation (Text → Image)

Image generation converts textual descriptions into images. While DALL·E 3 is used in this example, there are many other open-source or commercial options:

- **Stable Diffusion 3** ([Stability AI](https://stability.ai/stable-diffusion))
- **Flux.1-dev** ([Hugging Face repo](https://huggingface.co/black-forest-labs/FLUX.1-dev))
- **ComfyUI** for advanced image generation workflows ([GitHub](https://github.com/comfyanonymous/ComfyUI))
  - We had a ComfyUI expert come in a previous cohort and do a guest lecture. You can access that through [this link](https://us06web.zoom.us/rec/share/liE7XxnKleO7zAicH6FBaxTHc92CJSyV_tAlboDSVRJEImCCdMNwWe7lzSJC9hQ_.w3WbHJaa9_cBHFjL?startTime=1723568224000):
  - Passcode: \*XtL3Sb$
  - 10/10 no notes, you MUST watch this.

```python
# src/modal_app/main.py
@fastapi_app.post("/generate_image")
async def generate_image(request: ImageGenerationRequest):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=request.prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return {"image_url": response.data[0].url}
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return {"error": str(e)}, 500
```

**Note:** We’ve also seen big developments with [Lexica Aperture v4](https://twitter.com/sharifshameem/status/1760342835994439936) and open-source diffusion pipelines like [MLC (Machine Learning Compilation)](https://mlc.ai/) that help you run stable diffusion locally on consumer GPUs. [Like in your browser!](https://github.com/mlc-ai/web-stable-diffusion?tab=readme-ov-file#web-stable-diffusion)

### Image Analysis (Image → Understanding)

Image analysis can combine CLIP-based similarity scoring with large-vision-model understanding. Over the past year, CLIP variations have proliferated, and new vision-based LLMs (like GPT-4 Vision or Claude Vision) have emerged:

```python
# src/modal_app/main.py
@fastapi_app.post("/analyze_image_similarity")
async def analyze_image_similarity(request: ImageSimilarityRequest):
    # CLIP for numerical similarity
    model = SentenceTransformer('clip-ViT-B-32')
    # massage the image into the format the model wants
    image_response = urlopen(request.image_url)
    image = Image.open(BytesIO(image_response.read())).convert('RGB')
    # get the image and text embeddings
    img_emb = model.encode(image)
    text_emb = model.encode([request.prompt])
    # get the similarity between the image and text embeddings
    similarity = util.cos_sim(img_emb, text_emb)

    # Vision model for detailed analysis
    client = OpenAI()
    vision_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": request.image_url}}
            ]
        }]
    )
    return {
        "similarity_score": float(similarity[0][0]) * 100,
        "image_description": vision_response.choices[0].message.content
    }
```

Alternative implementations:

- For the vision/embeddings — the industry is really around CLIP and various fine-tuned versions of it. You can see all the different flavors [here](https://huggingface.co/openai/clip-vit-large-patch14).
- For vision: [Claude 3](https://docs.anthropic.com/en/docs/build-with-claude/vision) or [Gemini Pro Vision](https://ai.google.dev/gemini-api/docs/vision?lang=python)

### Text-to-Speech (Text → Speech)

The final component converts text back into speech:

```python
# src/modal_app/main.py
@fastapi_app.post("/text_to_speech")
async def text_to_speech(request: TextToSpeechRequest):
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=request.text
    )
    audio_base64 = base64.b64encode(response.content).decode('utf-8')
    return {"audio": audio_base64}
```

Alternative implementations:

- Open source: Our friends over at Modal wrote up a fairly recent article that answers this specific question more in-depth [here](https://modal.com/blog/open-source-tts)
  - Their TLDR:
    - If you need real-time: [Ultravox](hhttps://github.com/fixie-ai/ultravox)
    - If you only need English: [StyleTTS](https://github.com/yl4579/StyleTTS2?tab=readme-ov-file)
    - If you need it to run on-device: [VITS](https://github.com/jaywalnut310/vits)
- For Closed source: [ElevenLabs](https://elevenlabs.io/) or [Vapi](https://vapi.ai/)

## System Integration

The front end orchestrates these components using React and TypeScript. Below is a simplified flow. For full code, see the snippet that follows:

1. **Initialize audio recording** using `MediaRecorder`
2. **Send audio** to `/transcribe` endpoint
3. **Use transcript** to generate an image
4. **Analyze generated image** for similarity & description
5. **Convert analysis to speech** for final output

The key is maintaining state management for the entire pipeline while handling asynchronous operations effectively.

The only thing you should need to add is the progress bar. to install that run this in the `frontend_directory`:

```bash
bunx --bun shadcn@latest add progress
```

Here's all the code that we need:

```jsx
// frontend_service/src/App.tsx
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/toaster";
import { useToast } from "@/hooks/use-toast";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";

function MultiModalApp() {
  const { toast } = useToast();
  const modalUrl = import.meta.env.VITE_MODAL_URL;

  // State for microphone selection
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);

  // State for audio recording
  const [isRecording, setIsRecording] = useState(false);
  const [recorder, setRecorder] = useState<MediaRecorder | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);

  // Processing states
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  // Result states
  const [transcript, setTranscript] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [similarityScore, setSimilarityScore] = useState<number | null>(null);
  const [imageDescription, setImageDescription] = useState<string>("");
  const [descriptionAudio, setDescriptionAudio] = useState<string>("");

  // Handle requesting microphone permissions
  const handleRequestMicPermissions = async () => {
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      const devices = await navigator.mediaDevices.enumerateDevices();
      const microphones = devices.filter((d) => d.kind === "audioinput");
      setAudioDevices(microphones);

      if (microphones.length > 0) {
        setSelectedDeviceId(microphones[0].deviceId);
      } else {
        toast({
          title: "No Microphones",
          description: "No microphone devices were found.",
          variant: "destructive",
        });
      }
    } catch (err: any) {
      toast({
        title: "Permission Error",
        description:
          err.name === "NotAllowedError"
            ? "Microphone permission was denied."
            : `Error: ${err.message}`,
        variant: "destructive",
      });
      console.error("Error requesting mic permission:", err);
    }
  };

  // Handle recording start
  const handleStartRecording = async () => {
    try {
      if (!selectedDeviceId) {
        toast({
          title: "No Microphone",
          description: "Please select a microphone first.",
          variant: "destructive",
        });
        return;
      }

      const constraints = {
        audio: {
          deviceId: { exact: selectedDeviceId },
        },
      };

      const mimeType = "audio/webm; codecs=opus";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        toast({
          title: "Browser Not Supported",
          description: "Your browser doesn't support WebM with Opus codec.",
          variant: "destructive",
        });
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      const newRecorder = new MediaRecorder(stream, { mimeType });

      // Clear previous recording data
      setRecordedBlob(null);

      let chunks: Blob[] = [];
      newRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      newRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        setRecordedBlob(blob);
        stream.getTracks().forEach((track) => track.stop());
      };

      newRecorder.start(1000);
      setRecorder(newRecorder);
      setIsRecording(true);

      toast({
        title: "Recording Started",
        description: "Speak your prompt clearly into the microphone.",
      });
    } catch (err: any) {
      console.error("Error starting recording:", err);
      toast({
        title: "Recording Error",
        description: err.message,
        variant: "destructive",
      });
    }
  };

  // Handle recording stop
  const handleStopRecording = () => {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setIsRecording(false);
      toast({
        title: "Recording Complete",
        description: "You can now process your recording.",
      });
    }
  };

  // Process the full flow
  const handleProcessFlow = async () => {
    if (!recordedBlob) {
      toast({
        title: "No Recording",
        description: "Please record some audio first.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setProgress(0);

    try {
      // Step 1: Transcribe audio
      setCurrentStep("transcribing");
      setProgress(20);

      const formData = new FormData();
      formData.append("file", recordedBlob, "recording.webm");

      const transcriptResponse = await fetch(`${modalUrl}/transcribe`, {
        method: "POST",
        body: formData,
      });

      if (!transcriptResponse.ok) {
        const error = await transcriptResponse.json();
        throw new Error(error.detail || "Failed to transcribe audio");
      }

      const transcriptData = await transcriptResponse.json();
      setTranscript(transcriptData.transcript);
      setProgress(40);

      // Step 2: Generate image
      setCurrentStep("generating");
      const imageResponse = await fetch(`${modalUrl}/generate_image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: transcriptData.transcript }),
      });

      if (!imageResponse.ok) {
        const error = await imageResponse.json();
        throw new Error(error.detail || "Failed to generate image");
      }

      const imageData = await imageResponse.json();
      setImageUrl(imageData.image_url);
      setProgress(60);

      // Step 3: Analyze image similarity
      setCurrentStep("analyzing");
      const analysisResponse = await fetch(
        `${modalUrl}/analyze_image_similarity`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: transcriptData.transcript,
            image_url: imageData.image_url,
          }),
        },
      );

      if (!analysisResponse.ok) {
        const error = await analysisResponse.json();
        throw new Error(error.detail || "Failed to analyze image");
      }

      const analysisData = await analysisResponse.json();
      setSimilarityScore(analysisData.similarity_score);
      setImageDescription(analysisData.image_description);
      setProgress(80);

      // Step 4: Generate audio description
      setCurrentStep("speaking");
      if (analysisData.image_description) {
        const ttsResponse = await fetch(`${modalUrl}/text_to_speech`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: analysisData.image_description }),
        });

        if (!ttsResponse.ok) {
          const error = await ttsResponse.json();
          throw new Error(error.detail || "Failed to convert text to speech");
        }

        const ttsData = await ttsResponse.json();
        setDescriptionAudio(ttsData.audio);
      }

      setProgress(100);
      toast({
        title: "Processing Complete",
        description: "All steps have been completed successfully.",
      });
    } catch (error: any) {
      console.error("Processing error:", error);
      toast({
        title: "Processing Error",
        description: error.message || "An error occurred during processing",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
      setCurrentStep(null);
    }
  };

  const getStepDescription = () => {
    switch (currentStep) {
      case "transcribing":
        return "Transcribing your audio...";
      case "generating":
        return "Generating an image from your description...";
      case "analyzing":
        return "Analyzing the generated image...";
      case "speaking":
        return "Creating audio description...";
      default:
        return "";
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <Card className="mb-8">
        <CardContent className="pt-6">
          <h1 className="text-2xl font-bold mb-6">Multi-Modal AI Demo</h1>

          {/* Microphone Setup Section */}
          <div className="space-y-4 mb-8">
            <h2 className="text-xl font-semibold">Microphone Setup</h2>
            <Button
              variant="outline"
              onClick={handleRequestMicPermissions}
              className="w-full sm:w-auto"
            >
              Request Microphone Permissions
            </Button>
            <div className="mt-2">
              <label
                htmlFor="mic-select"
                className="block text-sm text-gray-600 mb-2"
              >
                Choose Microphone:
              </label>
              <select
                id="mic-select"
                className="w-full rounded-md border border-gray-300 shadow-sm p-2"
                value={selectedDeviceId ?? ""}
                onChange={(e) => setSelectedDeviceId(e.target.value)}
              >
                <option value="">Select a microphone...</option>
                {audioDevices.map((device) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label ||
                      `Microphone ${device.deviceId.slice(0, 8)}`}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Recording Controls */}
          <div className="space-y-4 mb-8">
            <h2 className="text-xl font-semibold">Record Your Prompt</h2>
            <div className="flex gap-2 flex-wrap">
              {!isRecording ? (
                <Button
                  onClick={handleStartRecording}
                  disabled={!selectedDeviceId || isProcessing}
                  className="w-full sm:w-auto"
                >
                  Start Recording
                </Button>
              ) : (
                <Button
                  variant="destructive"
                  onClick={handleStopRecording}
                  className="w-full sm:w-auto"
                >
                  Stop Recording
                </Button>
              )}

              {recordedBlob && (
                <>
                  <div className="w-full">
                    <audio
                      controls
                      src={URL.createObjectURL(recordedBlob)}
                      className="w-full mt-2"
                    />
                  </div>
                  <Button
                    onClick={handleProcessFlow}
                    disabled={isRecording || isProcessing}
                    className="w-full sm:w-auto"
                  >
                    Process Recording
                  </Button>
                </>
              )}
            </div>
          </div>

          {/* Processing Progress */}
          {isProcessing && (
            <div className="space-y-2 mb-8">
              <div className="flex justify-between text-sm">
                <span>{getStepDescription()}</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} className="w-full" />
            </div>
          )}

          {/* Results Display */}
          {transcript && (
            <div className="space-y-4 mb-8">
              <h2 className="text-xl font-semibold">Results</h2>

              <div className="space-y-2">
                <h3 className="font-semibold">Your Prompt:</h3>
                <p className="text-gray-700 bg-gray-50 p-4 rounded-lg">
                  {transcript}
                </p>
              </div>

              {imageUrl && (
                <div className="space-y-2">
                  <h3 className="font-semibold">Generated Image:</h3>
                  <img
                    src={imageUrl}
                    alt="AI Generated"
                    className="w-full max-w-2xl rounded-lg shadow-lg"
                  />
                  {similarityScore !== null &&
                    typeof similarityScore === "number" && (
                      <p className="text-sm text-gray-600">
                        Similarity to prompt: {similarityScore.toFixed(1)}%
                      </p>
                    )}
                </div>
              )}

              {imageDescription && (
                <div className="space-y-2">
                  <h3 className="font-semibold">AI Vision Analysis:</h3>
                  <p className="text-gray-700 bg-gray-50 p-4 rounded-lg">
                    {imageDescription}
                  </p>
                  {descriptionAudio && (
                    <div className="mt-4">
                      <h4 className="font-semibold mb-2">Audio Description:</h4>
                      <audio
                        controls
                        src={`data:audio/mp3;base64,${descriptionAudio}`}
                        className="w-full"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Toaster />
    </div>
  );
}

export default MultiModalApp;
```

Let's examine each major section:

### State Management and Initial Setup

```typescript
const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
const [isRecording, setIsRecording] = useState(false);
const [recorder, setRecorder] = useState<MediaRecorder | null>(null);
```

The application uses React's useState hooks to manage several pieces of state. These include tracking available audio devices, which device is selected, recording status, and the actual MediaRecorder instance. This separation of concerns helps manage the complex state needed for audio recording.

### Microphone Permissions and Setup

The `handleRequestMicPermissions` function manages the critical first step of getting user permission to access their microphone:

```typescript
const handleRequestMicPermissions = async () => {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
    const devices = await navigator.mediaDevices.enumerateDevices();
    const microphones = devices.filter((d) => d.kind === "audioinput");
    setAudioDevices(microphones);
    // ...
  }
```

This function uses the Web Audio API to:

1. Request microphone access
2. Enumerate available audio devices
3. Filter for just the microphone inputs
4. Store them in state for the user to select from

### Recording Functionality

The recording process is handled by two main functions:

`handleStartRecording`:

```typescript
const handleStartRecording = async () => {
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  const newRecorder = new MediaRecorder(stream, { mimeType });

  let chunks: Blob[] = [];
  newRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      chunks.push(event.data);
    }
  };
```

This function:

1. Gets an audio stream from the selected microphone
2. Creates a MediaRecorder instance
3. Sets up storage for the recorded audio chunks
4. Starts the recording process

The complementary `handleStopRecording` function stops the recording and makes the audio available for playback and processing.

### The Processing Pipeline

The most complex part is the `handleProcessFlow` function, which orchestrates the multi-modal transformations. We've omitted some of the code that logs out errors to clean up this snippet here so you can see where the bulk of the work is happening:

```typescript
const handleProcessFlow = async () => {
  // Step 1: Transcribe audio
  const transcriptResponse = await fetch(`${modalUrl}/transcribe`...);

  // Step 2: Generate image
  const imageResponse = await fetch(`${modalUrl}/generate_image`...);

  // Step 3: Analyze image similarity
  const analysisResponse = await fetch(`${modalUrl}/analyze_image_similarity`...);

  // Step 4: Generate audio description
  const ttsResponse = await fetch(`${modalUrl}/text_to_speech`...);
}
```

This function demonstrates a modern AI pipeline where:

1. Audio is converted to text using speech recognition
2. That text is used to generate an image
3. The image is analyzed for similarity to the original prompt
4. A description of the image is generated and converted back to speech

### Error Handling

Throughout the code, there's robust error handling using try-catch blocks and the toast notification system:

```typescript
} catch (error: any) {
  console.error("Processing error:", error);
  toast({
    title: "Processing Error",
    description: error.message || "An error occurred during processing",
    variant: "destructive",
  });
}
```

This ensures users always know what's happening, whether successful or not.

This approach keeps the pipeline simple but can be expanded with agent-based orchestrators like [LangChain Agents](https://github.com/langchain-ai/langchain), [DSPy modules](https://github.com/stanfordnlp/dspy), or advanced RAG frameworks (Retrieval-Augmented Generation) to store longer dialogues or retrieve user context from multiple data sources. As you can see, while we aren't using agents yet (that's next week!) you can see how managing so many different data flows can get complicated.

## Conclusion

The rapid evolution of multi-modal AI has opened the door to novel products and research. By combining speech recognition, image generation, image analysis, and text-to-speech in a single pipeline, you can deliver experiences that feel increasingly human—understanding user speech, responding with synthetic voices, generating relevant images, and explaining them.

Over the past year, many new open-source solutions (like [Mistral](https://mistral.ai/), [DeepSeekCoder](https://x.com/teortaxestex/status/1752177206283964813), or various [TTS frameworks](https://github.com/fixie-ai/ultravox)) have appeared, giving developers greater freedom to experiment. On the closed-source side, we have powerful but tightly controlled offerings from companies like OpenAI, Anthropic, Stability AI, Google, and others—each pushing the boundaries of multi-modal functionality.

As you continue building out your own multi-modal AI systems, keep a careful eye on:

1. **Data pipelines**: Ensuring your input transformations are correct and maintaining consistent data quality.
2. **Model selection and orchestration**: Weigh open-source vs. closed-source offerings; consider new model architectures like Mixture-of-Experts or specialized modules for images, text, or speech.
3. **User experience**: From error handling and progress feedback (like the React `Progress` component above) to final text-to-speech output, every step should be cohesive and user-friendly.
4. **Ethical and privacy considerations**: Especially with user-generated audio or personal images, ensure compliance with local regulations, and consider on-device solutions when data privacy is crucial.

Multi-modal AI isn’t just about plugging together random models. It’s about creating a seamless collaboration of specialized components that augment and complement each other. By following the blueprint of a pipeline architecture—and leveraging the wide range of open-source resources that have matured this past year—you can craft robust, cutting-edge applications that truly harness the power of multiple modalities.
