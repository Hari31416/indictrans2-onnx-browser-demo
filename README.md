# IndicTrans2 ONNX Browser Demo

This is a web playground that runs optimized IndicTrans2 translation models entirely in the browser using ONNX Runtime Web and Hugging Face Transformers.js. All execution happens on your GPU or CPU without server-side APIs.

For detailed model conversion steps, export scripts, and implementation details, please refer to the main [GitHub Repository](https://github.com/Hari31416/indictrans2-onnx-export).

## Model Collections

The following Hugging Face Collections host the exported ONNX model variants used in this demo:

- [200M/320M Models Collection](https://huggingface.co/collections/hari31416/indictrans2-onnx-exports)
- [1B Models Collection](https://huggingface.co/collections/hari31416/indictrans2-1b-onnx-exports)

## Features

- **Local Execution**: Models run fully client-side via WebGPU or WebAssembly.
- **On-Demand Loading**: Select and download only the configuration, precision, and model size you want to use.
- **Caching Support**: Models are stored in browser Cache Storage so that subsequent runs load instantly without re-downloading.
- **Pre/Post-Processing**: Built-in script-transliteration pipelines for non-Devanagari scripts to optimize accuracy.
- **Performance Benchmarking**: Real-time logging of model load latency, TTFT (time-to-first-token), and tokens/sec generation speed.

## File Structure

The project has a modular layout:

- `index.html`: Clean HTML structure containing the main UI layouts.
- `style.css`: Styling configurations and glowing theme details.
- `transliterate.js`: Script range mappings and pre/post-processing transliteration algorithms.
- `translator.js`: ONNX Runtime session initializer, downloading utility, and translation/generation logic.
- `app.js`: Application controller connecting dropdowns, loaders, text fields, and performance metrics.

## Running Locally

To run this demo in your local browser, start a simple HTTP server in the demo directory:

```bash
python3 -m http.server 8005
```

Open `http://localhost:8005` in your browser.

## Deploying to Hugging Face Spaces

This demo is designed to run directly as a static Hugging Face Space.

To deploy it:

1. Create a new Space on Hugging Face.
2. Select **Static** HTML as the Space SDK.
3. Clone the Space repository and copy the contents of this folder (`index.html`, `style.css`, `transliterate.js`, `translator.js`, `app.js`) directly into the repository root.
4. Commit and push the changes. Hugging Face will build and host the space automatically.
