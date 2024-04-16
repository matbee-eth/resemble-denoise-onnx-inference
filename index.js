import ort from 'onnxruntime-node';
import { promises as fs } from 'fs';
import shortTimeFT, { istft as inverseSTFT } from 'stft';
import audioenc from 'audio-encoder';
import AudioBuffer from "audio-buffer"
import { AudioContext } from 'web-audio-api';
import { spawn } from 'child_process';
import path from 'path';
import os from 'os';

async function processAudio(audioData, onnxSession) {
    const hopSize = 420;
    const windowSize = 4 * hopSize;
    const fftSize = windowSize;
    const expectedMagDim = windowSize / 2 + 1;
    const frequencyBins = fftSize / 2 + 1;

    console.log("audioData.length", audioData.length)
    // Pad the audio data if necessary
    // const paddedAudioData = padAudioData(audioData, fftSize);
    const paddedAudioData = audioData;
    // console.log("paddedAudioData", paddedAudioData.length, paddedAudioData)

    // Perform STFT on the padded audio data
    const stftData = await performSTFT(paddedAudioData.getChannelData(0), fftSize, hopSize);
    // console.log("stftData", stftData.length, stftData[0].re.slice(0, 20))

    // Process each frame with the ONNX model
    const processedFrames = await processAudioWithONNX(stftData, onnxSession, expectedMagDim, hopSize);
    console.log("processedFrames", processedFrames.length, processedFrames.slice(0, 2))

    // Perform ISTFT on processed frames
    const processedAudio = await performISTFT(processedFrames, fftSize, hopSize);
    console.log("processedAudio", processedAudio.length)

    return processedAudio;
}

function padAudioData(audioData, fftSize) {
    const paddingSize = fftSize - (audioData.length % fftSize);
    const reflectedPadding = Math.floor(paddingSize / 2);
    const remainingPadding = paddingSize - reflectedPadding;

    const audioContext = new AudioContext();
    const paddingAudioBuffer = audioContext.createBuffer(1, audioData.length + paddingSize, 44100);

    const reflectedAudioBuffer = new AudioBuffer({
        length: audioData.length + paddingSize,
        numberOfChannels: 1,
        sampleRate: 44100,
    });
    const paddedAudioData = new Float32Array(audioData.length + paddingSize);
    paddedAudioData.set(audioData.slice(-reflectedPadding)._data[0], 0);
    paddedAudioData.set(audioData, reflectedPadding);
    paddedAudioData.set(audioData.slice(0, remainingPadding)._data[0], audioData.length + reflectedPadding);
    reflectedAudioBuffer.copyToChannel(paddedAudioData, 0);

    paddingAudioBuffer.getChannelData(0).set(reflectedAudioBuffer.getChannelData(0));
    return paddingAudioBuffer;
}

async function performSTFT(audioData, fftSize, hopSize) {
    return new Promise((resolve) => {
        const frames = [];
        const numFrames = Math.ceil((audioData.length - hopSize) / hopSize) + 1;

        for (let i = 0; i < numFrames; i++) {
            const start = i * hopSize;
            const end = start + hopSize;
            let frame = audioData.slice(start, end);
            // console.log("frame", frame.length)
            if (frame.length < hopSize) {
                // Pad the last frame with zeros if it's shorter than fftSize
                const paddedFrame = new Float32Array(hopSize);
                paddedFrame.set(frame);
                frame = paddedFrame;
            }

            // Apply the window function to the frame
            const windowedFrame = applyWindowFunction(frame, hopSize, hannWindowAnalysis);

            // Compute STFT for the windowed frame
            const stft = shortTimeFT(1, fftSize, (re, im) => {
                // console.log("re", re.length, "im", im.length)
                // console.log("re", re.slice(0, 2), "im", im.slice(0, 2))
                frames.push({ re: re, im: im });
                // console.log("frames", frames.length, frames[frames.length - 1])
            }, {
                hopSize: hopSize,
            });
            // console.log("Executing STFT", windowedFrame.length)
            stft(windowedFrame);
        }

        resolve(frames);
    });
}

function performISTFT(frames, fftSize, hopSize) {
    console.info("Performing ISTFT", frames.length, fftSize, hopSize, frames.slice(0, 100));
    return new Promise((resolve) => {
        const numFrames = frames.length;
        const reconstructedSignal = new Float32Array(numFrames * hopSize + fftSize - hopSize);
        let offset = 0;

        const istft = inverseSTFT(fftSize, (signal) => {
            console.log("signal", signal.length);

            // Apply a windowing function (e.g., Hann window) to the reconstructed signal
            const windowFunction = hannWindow(signal.length);
            const windowedSignal = signal.map((sample, index) => sample * windowFunction[index]);

            // Overlap-add the windowed signal
            for (let i = 0; i < windowedSignal.length; i++) {
                reconstructedSignal[offset + i] += windowedSignal[i];
            }
            offset += hopSize;

            // Check if all frames have been processed
            if (offset >= numFrames * hopSize) {
                // Find the absolute maximum value of the reconstructed signal
                let maxAbsValue = 0;
                for (let i = 0; i < reconstructedSignal.length; i++) {
                    const absValue = Math.abs(reconstructedSignal[i]);
                    if (absValue > maxAbsValue) {
                        maxAbsValue = absValue;
                    }
                }
                const absMax = Math.max(1e-7, maxAbsValue);

                // Normalize the reconstructed signal
                const normalizedSignal = reconstructedSignal.map((sample) => sample / absMax);

                console.log("finalSignal", normalizedSignal.length);
                resolve(normalizedSignal);
            }
        }, {
            hopSize: hopSize,
        });

        // Process each frame with ISTFT
        frames.forEach((frame) => {
            const { re, im } = frame;
            console.log("re", re.length, "im", im.length);
            istft(re, im);
        });
    });
}

// Helper function to generate a Hann window
function hannWindow(length) {
    const window = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
    }
    return window;
}

function applyWindowFunction(frame, hopSize, windowFunc) {
    const windowedFrame = new Float32Array(hopSize);
    for (let i = 0; i < hopSize; i++) {
        windowedFrame[i] = frame[i] * windowFunc(i / hopSize);
    }
    return windowedFrame;
}

function hannWindowAnalysis(t) {
    return 0.5 * (1.0 - Math.cos(2.0 * Math.PI * t));
}

async function processAudioWithONNX(audioData, onnxSession, expectedMagDim, hopSize) {
    const frames = [];

    for (let { re, im } of audioData) {
        const magData = new Float32Array(expectedMagDim);
        const phiData = new Float32Array(expectedMagDim);

        // Convert complex to magnitude and phase
        for (let j = 0; j < re.length; j++) {
            // console.log("re[j]", re[j], "im[j]", im[j])
            magData[j] = Math.sqrt(re[j] ** 2 + im[j] ** 2);
            phiData[j] = Math.atan2(im[j], re[j]);
        }

        const cosData = new Float32Array(magData.length);
        const sinData = new Float32Array(magData.length);

        // Prepare cos and sin data based on phase information
        for (let j = 0; j < phiData.length; j++) {
            cosData[j] = Math.cos(phiData[j]);
            sinData[j] = Math.sin(phiData[j]);
        }

        // Preparing the tensors for ONNX model input
        const magTensor = new ort.Tensor('float32', magData, [1, expectedMagDim, 1]);
        const cosTensor = new ort.Tensor('float32', cosData, [1, expectedMagDim, 1]);
        const sinTensor = new ort.Tensor('float32', sinData, [1, expectedMagDim, 1]);

        // console.log("magTensor", magData.slice(0, 10))

        const feeds = { mag: magTensor, cos: cosTensor, sin: sinTensor };
        const results = await onnxSession.run(feeds);

        // Assuming the ONNX model outputs tensors named 'out_mag', 'out_cos', and 'out_sin'
        const outMagData = results.out_mag.data;
        const outCosData = results.out_cos.data;
        const outSinData = results.out_sin.data;

        // console.log("outMagData", outMagData.slice(0, 10))
        // console.log("outCosData", outCosData.slice(0, 10))
        // console.log("outSinData", outSinData.slice(0, 10))

        // Reconstructing the processed complex signal
        const processedRe = new Float32Array(hopSize);
        const processedIm = new Float32Array(hopSize);

        for (let j = 0; j < hopSize; j++) {
            processedRe[j] = outMagData[j] * outCosData[j];
            processedIm[j] = outMagData[j] * outSinData[j];
            // console.log("processedRe[j]", processedRe[j], "processedIm[j]", processedIm[j])
        }

        frames.push({ re: processedRe, im: processedIm });
    }

    return frames;
}

async function run(onnxSession, audio, sampleRate, numChannels) {
    // console.log('Starting audio processing');
    // console.log('Input audio length:', audio.length);
    // console.log('Sample rate:', sampleRate);
    // console.log('Number of channels:', numChannels);

    // if (sampleRate !== 44100 || numChannels !== 1) {
    //     const resampledAudioBuffer = await resampleAudioBuffer(audio, 44100);
    //     audio = resampledAudioBuffer;
    //     console.log('Resampled audio length');
    // }
    // Normalize audio to [-1, 1]
    console.log('Normalizing audio');
    console.log("audio", audio.slice(0, 2), "type", typeof audio);

    // Pad audio to a minimum length
    const minLengthSec = 1.0;
    const minLengthSamples = Math.ceil(minLengthSec * 44100);
    console.log('Padding audio to minimum length of', minLengthSamples, 'samples', typeof audio, "current length", audio.length, "sample", audio.slice(0, 2));
    const audioContext = new AudioContext();
    const paddedAudio = audioContext.createBuffer(1, Math.max(minLengthSamples, audio.length), 44100);
    const channelData = paddedAudio.getChannelData(0);
    channelData.set(audio._data[0]);

    let maxAbsValue = 0;
    for (let i = 0; i < audio._data[0].length; i++) {
        const absValue = Math.abs(audio._data[0][i]);
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue;
        }
    }
    // Normalize audio
    console.log("maxAbsValue", maxAbsValue)
    if (maxAbsValue > 0) {
        const scaleFactor = 1 / maxAbsValue;
        // Normalize audio
        for (let i = 0; i < channelData.length; i++) {
            channelData[i] = audio._data[0][i] * scaleFactor;
        }
    } else {
        channelData.set(audio._data[0]);
    }

    console.log('Padded audio to minimum length of', minLengthSamples, 'samples', "current length", paddedAudio.length, "sample", paddedAudio._data[0].slice(0, 2));
    // await playAudio(paddedAudio);
    console.log('Padded audio length:', paddedAudio.length, paddedAudio);

    // Process the entire audio
    console.log('Processing audio');
    const processedAudio = await processAudio(paddedAudio, onnxSession);
    console.log('Processed audio length:', processedAudio.length, processedAudio.slice(0, 2));
    // Normalize the processed audio
    console.log('Normalizing processed audio');
    const processedAudioContext = new AudioContext();
    const processedAudioBuffer = processedAudioContext.createBuffer(1, Math.max(minLengthSamples, processedAudio.length), 44100);
    processedAudioBuffer.getChannelData(0).set(processedAudio);

    console.log('Audio processing completed');
    return [processedAudioBuffer, 44100];
}

async function loadAudio(path) {
    const buffer = await fs.readFile(path);

    return decodeAudioData(buffer)
}

function decodeAudioData(audioBuffer) {
    // console.log("Decoding audio data", audioBuffer, audioBuffer?.length)
    const audioContext = new AudioContext();
    return new Promise((resolve, reject) => {
        // console.info("audioBuffer", audioBuffer)
        audioContext.decodeAudioData(audioBuffer, (audioBuffer) => {
            if (!audioBuffer) {
                reject(new Error('Failed to decode audio data'));
                return;
            }

            // Convert from stereo to mono
            const numChannels = 1;
            const sr = audioBuffer.sampleRate;
            const monoBuffer = convertToMono(audioBuffer);

            resolve([monoBuffer, sr, numChannels]);
        }, (error) => {
            reject(error);
        });
    });
}

function convertToMono(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    if (numChannels === 1) {
        return audioBuffer; // Already mono
    }

    const monoChannelData = audioBuffer.getChannelData(0); // Use only the first channel

    const audioContext = new AudioContext();
    const monoAudioBuffer = audioContext.createBuffer(1, audioBuffer.length, audioBuffer.sampleRate);
    monoAudioBuffer.getChannelData(0).set(monoChannelData);

    return monoAudioBuffer;
}

async function writeWav(audioData, filePath) {
    return new Promise((resolve, reject) => {
        console.info("Writing wav", audioData.length)
        audioenc(audioData, 'WAV', null, async function onComplete(blob) {
            await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
            resolve();
        });
    });
}

async function bufferToAudioBuffer(buffer, sampleRate) {
    console.log("bufferToAudioBuffer", buffer.slice(0, 2), buffer.length, sampleRate)
    const audioContext = new AudioContext();
    const audioBuffer = audioContext.createBuffer(1, buffer.length, sampleRate);
    console.log("audioBuffer", audioBuffer.length, audioBuffer.sampleRate)
    const channelData = audioBuffer.getChannelData(0);
    channelData.set(buffer);
    console.log("channelData", channelData.slice(0, 2), channelData.length)
    return new Promise((resolve, reject) => {
        audioenc(audioBuffer, 'WAV', null, async function onComplete(blob) {
            resolve(Buffer.from(await blob.arrayBuffer()));
        });
    });
}

const audiopath = './untitled.wav';

async function playAudio(audioPathOrBuffer) {
    try {
        let audioBuffer = audioPathOrBuffer;
        // If audioBuffer is already a buffer, do nothing
        // No action needed when audioBuffer is already a buffer
        if (typeof audioPathOrBuffer === 'string') {
            audioBuffer = await fs.readFile(audioPathOrBuffer);
            console.log("LOADED AUDIO BUFFER", audioBuffer);
        } else if (audioPathOrBuffer instanceof Float32Array) {
            console.log("LOADING FLOAT32ARRAY AUDIO BUFFER", audioBuffer.slice(0, 2), audioBuffer.length);
            audioBuffer = await bufferToAudioBuffer(audioBuffer, 44100);
            console.log("LOADED AUDIO BUFFER", audioBuffer.slice(0, 2), audioBuffer.length);
        }
        console.log("Playing audio", audioPathOrBuffer.slice(0, 2), audioBuffer.length);

        const tempDir = os.tmpdir();
        const tempFilePath = path.join(tempDir, 'temp.wav');
        console.log("Writing file", tempFilePath, audioBuffer.slice(0, 2), audioBuffer.length);
        await writeWav(audioBuffer, tempFilePath);
        const stats = await fs.stat(tempFilePath);
        console.log('File size:', stats.size);
        console.log(audioBuffer.length, tempFilePath)
        await new Promise((resolve, reject) => {
            const ffplay = spawn('ffplay', ['-nodisp', '-autoexit', '-f', 'wav', '-t', 2, tempFilePath]);

            ffplay.on('close', async (code) => {
                await fs.unlink(tempFilePath);
                if (code === 0) {
                    console.log('Audio playback complete');
                    resolve();
                } else {
                    reject(new Error(`FFplay exited with code ${code}`));
                }
            });

            ffplay.on('error', (error) => {
                fs.unlink(tempFilePath);
                reject(error);
            });
        });
    } catch (error) {
        console.error('Error playing audio:', error);
    }
}


async function main() {
    try {
        const audioFilePath = audiopath;

        const [audio, sr, numChannels] = await loadAudio(audioFilePath);
        const outputPath = 'resampled-js.wav';
        await writeWav(audio, outputPath);
        console.info("audio", audio.slice(0, 2))
        // await playAudio(audio);
        console.log('Resampled playing:::');

        const opts = {
            executionProviders: ['cpu'],
            interOpNumThreads: 4,
            intraOpNumThreads: 4,
            logSeverityLevel: 4,
        };

        const session = await ort.InferenceSession.create(
            'denoiser.onnx',
            opts,
        );

        const start = Date.now();
        const [processedAudioData, bitrate] = await run(session, audio, sr, numChannels);
        console.log(`Ran in ${(Date.now() - start) / 1000}s`);
        console.log(`processedAudioData ${processedAudioData.length}`, processedAudioData)
        const outputFilePath = './denoiser_output.wav';
        await playAudio(processedAudioData)
        await writeWav(processedAudioData, outputFilePath);

        const stats = await fs.stat(outputFilePath);
        const fileSizeInBytes = stats.size;
        const fileSizeInKB = fileSizeInBytes / 1024;
        const fileSizeInMB = fileSizeInKB / 1024;

        console.log('File Size:', fileSizeInBytes, 'bytes');
        console.log('File Size:', fileSizeInKB.toFixed(2), 'KB');
        console.log('File Size:', fileSizeInMB.toFixed(2), 'MB');
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
setInterval(function () {
    console.log("timer that keeps nodejs processing running");
}, 1000 * 60 * 60);