const faceapi = window.faceapi;
const ort = window.ort;

if (!faceapi) {
	throw new Error("face-api.js non caricato dal CDN.");
}

if (!ort) {
	throw new Error("onnxruntime-web non caricato dal CDN.");
}

const imageInput = document.getElementById("imageInput");
const runBtn = document.getElementById("runBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");
const inputMeta = document.getElementById("inputMeta");
const outputMeta = document.getElementById("outputMeta");
const strengthSlider = document.getElementById("strengthSlider");
const strengthValue = document.getElementById("strengthValue");
const promptInput = document.getElementById("promptInput");
const inputCanvas = document.getElementById("inputCanvas");
const outputCanvas = document.getElementById("outputCanvas");
const modelProgressWrap = document.getElementById("modelProgressWrap");
const modelProgressBar = document.getElementById("modelProgressBar");
const modelProgressText = document.getElementById("modelProgressText");
const modelProgressDetail = document.getElementById("modelProgressDetail");
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });
const outputCtx = outputCanvas.getContext("2d", { willReadFrequently: true });

const FACE_MODEL_URL = "/models/face-api";
const FACE_INPUT_SIZE = 512;
const FACE_PADDING_RATIO = 0.28;

const SD_MODEL_SOURCES = {
	text_encoder: [
		"/models/sd-turbo/text_encoder/model.onnx",
		"https://huggingface.co/microsoft/sd-turbo-webnn/resolve/main/text_encoder/model.onnx",
		"https://huggingface.co/webnn/sd-turbo-webnn/resolve/main/text_encoder/model.onnx",
	],
	unet: [
		"/models/sd-turbo/unet/model.onnx",
		"https://huggingface.co/microsoft/sd-turbo-webnn/resolve/main/unet/model.onnx",
		"https://huggingface.co/webnn/sd-turbo-webnn/resolve/main/unet/model.onnx",
	],
	vae_decoder: [
		"/models/sd-turbo/vae_decoder/model.onnx",
		"https://huggingface.co/microsoft/sd-turbo-webnn/resolve/main/vae_decoder/model.onnx",
		"https://huggingface.co/webnn/sd-turbo-webnn/resolve/main/vae_decoder/model.onnx",
	],
	vae_encoder: [
		"/models/sd-turbo/vae_encoder/model.onnx",
		"https://huggingface.co/eyaler/sd-turbo-webnn/resolve/main/vae_encoder/model.onnx",
	],
};

let loadedImage = null;
let sdSessions = null;
let sdTurboReady = false;

// -------------------------
// helpers
// -------------------------
function setStatus(msg) {
	statusEl.textContent = msg;
}

function clamp(value, min, max) {
	return Math.max(min, Math.min(max, value));
}

function randomInt(max) {
	return Math.floor(Math.random() * max);
}

function setCanvasSizeToImage(canvas, ctx, img) {
	const width = img.naturalWidth || img.width;
	const height = img.naturalHeight || img.height;
	canvas.width = width;
	canvas.height = height;
	ctx.clearRect(0, 0, width, height);
	ctx.drawImage(img, 0, 0, width, height);
}

function copyCanvas(src, dst, dstCtx) {
	dst.width = src.width;
	dst.height = src.height;
	dstCtx.clearRect(0, 0, dst.width, dst.height);
	dstCtx.drawImage(src, 0, 0);
}

function loadImageFromFile(file) {
	return new Promise((resolve, reject) => {
		const img = new Image();
		img.onload = () => resolve(img);
		img.onerror = reject;
		img.src = URL.createObjectURL(file);
	});
}

function supportsWebGPU() {
	return typeof navigator !== "undefined" && !!navigator.gpu;
}

function formatBytes(bytes) {
	if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
	const units = ["B", "KB", "MB", "GB"];
	let i = 0;
	let value = bytes;
	while (value >= 1024 && i < units.length - 1) {
		value /= 1024;
		i++;
	}
	return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[i]}`;
}

function resetModelProgressUI() {
	modelProgressWrap.style.display = "none";
	modelProgressBar.style.width = "0%";
	modelProgressText.textContent = "0%";
	modelProgressDetail.textContent = "In attesa...";
}

function updateModelProgressUI(progressState) {
	const entries = Object.entries(progressState.components);

	const totalLoaded = entries.reduce((sum, [, c]) => sum + (c.loaded || 0), 0);
	const totalKnown = entries.reduce((sum, [, c]) => sum + (c.total || 0), 0);

	let percent = 0;
	if (totalKnown > 0) {
		percent = Math.max(
			0,
			Math.min(100, Math.round((totalLoaded / totalKnown) * 100)),
		);
	}

	modelProgressWrap.style.display = "block";
	modelProgressBar.style.width = `${percent}%`;
	modelProgressText.textContent = `${percent}%`;

	const lines = entries.map(([name, c]) => {
		const loadedTxt = formatBytes(c.loaded || 0);
		const totalTxt = c.total ? formatBytes(c.total) : "sconosciuto";
		const icon =
			c.status === "done"
				? "✅"
				: c.status === "error"
					? "❌"
					: c.status === "loading"
						? "⬇️"
						: "⏳";

		const itemPercent =
			c.total && c.total > 0
				? ` (${Math.round((c.loaded / c.total) * 100)}%)`
				: "";

		return `${icon} ${name}: ${loadedTxt} / ${totalTxt}${itemPercent}`;
	});

	modelProgressDetail.textContent = lines.join("\n");
}

async function createSessionWithFallback(urls, sessionOptions, label) {
	let lastError = null;

	for (const url of urls) {
		try {
			console.log(`[PATCH 2B] trying ${label}:`, url);
			const session = await ort.InferenceSession.create(url, sessionOptions);
			console.log(`[PATCH 2B] loaded ${label}:`, url);
			return session;
		} catch (err) {
			console.warn(`[PATCH 2B] failed ${label}:`, url, err);
			lastError = err;
		}
	}

	throw lastError ?? new Error(`No valid source found for ${label}`);
}

// -------------------------
// runtime bootstrap
// -------------------------
async function initFaceDetection() {
	await faceapi.nets.ssdMobilenetv1.loadFromUri(
		`${FACE_MODEL_URL}/ssd_mobilenetv1`,
	);
}

async function fetchWithProgress(url, componentName, progressState) {
	const res = await fetch(url, { cache: "force-cache" });

	if (!res.ok) {
		throw new Error(`HTTP ${res.status} ${res.statusText}`);
	}

	const total = Number(res.headers.get("content-length") || 0);

	progressState.components[componentName] = {
		loaded: 0,
		total,
		status: "loading",
	};
	updateModelProgressUI(progressState);

	if (!res.body) {
		const blob = await res.blob();
		progressState.components[componentName].loaded = blob.size;
		progressState.components[componentName].total = blob.size;
		progressState.components[componentName].status = "done";
		updateModelProgressUI(progressState);
		return blob;
	}

	const reader = res.body.getReader();
	const chunks = [];
	let loaded = 0;

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;

		chunks.push(value);
		loaded += value.byteLength;

		progressState.components[componentName].loaded = loaded;
		updateModelProgressUI(progressState);
	}

	const blob = new Blob(chunks, { type: "application/octet-stream" });

	progressState.components[componentName].loaded = blob.size;
	progressState.components[componentName].total = total || blob.size;
	progressState.components[componentName].status = "done";
	updateModelProgressUI(progressState);

	return blob;
}

async function createSessionWithProgress(
	urls,
	sessionOptions,
	componentName,
	progressState,
) {
	let lastError = null;

	for (const url of urls) {
		try {
			console.log(`[PATCH 2B] trying ${componentName}:`, url);

			progressState.components[componentName] = {
				loaded: 0,
				total: 0,
				status: "loading",
			};
			updateModelProgressUI(progressState);

			const blob = await fetchWithProgress(url, componentName, progressState);
			const objectUrl = URL.createObjectURL(blob);

			try {
				const session = await ort.InferenceSession.create(
					objectUrl,
					sessionOptions,
				);
				console.log(`[PATCH 2B] loaded ${componentName}:`, url);
				return session;
			} finally {
				URL.revokeObjectURL(objectUrl);
			}
		} catch (err) {
			console.warn(`[PATCH 2B] failed ${componentName}:`, url, err);
			progressState.components[componentName] = {
				...(progressState.components[componentName] || {}),
				status: "error",
			};
			updateModelProgressUI(progressState);
			lastError = err;
		}
	}

	throw lastError ?? new Error(`No valid source found for ${componentName}`);
}

async function initSdTurboSessions() {
	if (!supportsWebGPU()) {
		throw new Error("WebGPU non disponibile nel browser.");
	}

	ort.env.logLevel = "warning";

	const sessionOptions = {
		executionProviders: ["webgpu"],
		graphOptimizationLevel: "all",
	};

	const progressState = {
		components: {
			text_encoder: { loaded: 0, total: 0, status: "pending" },
			unet: { loaded: 0, total: 0, status: "pending" },
			vae_decoder: { loaded: 0, total: 0, status: "pending" },
			vae_encoder: { loaded: 0, total: 0, status: "pending" },
		},
	};

	resetModelProgressUI();
	updateModelProgressUI(progressState);

	// sequenziale = barra leggibile e stato chiaro
	const textEncoder = await createSessionWithProgress(
		SD_MODEL_SOURCES.text_encoder,
		sessionOptions,
		"text_encoder",
		progressState,
	);

	const unet = await createSessionWithProgress(
		SD_MODEL_SOURCES.unet,
		sessionOptions,
		"unet",
		progressState,
	);

	const vaeDecoder = await createSessionWithProgress(
		SD_MODEL_SOURCES.vae_decoder,
		sessionOptions,
		"vae_decoder",
		progressState,
	);

	const vaeEncoder = await createSessionWithProgress(
		SD_MODEL_SOURCES.vae_encoder,
		sessionOptions,
		"vae_encoder",
		progressState,
	);

	sdSessions = { textEncoder, unet, vaeDecoder, vaeEncoder };

	console.log("[PATCH 2B] SD sessions ready", {
		textEncoderInputs: textEncoder.inputNames,
		textEncoderOutputs: textEncoder.outputNames,
		unetInputs: unet.inputNames,
		unetOutputs: unet.outputNames,
		vaeDecoderInputs: vaeDecoder.inputNames,
		vaeDecoderOutputs: vaeDecoder.outputNames,
		vaeEncoderInputs: vaeEncoder.inputNames,
		vaeEncoderOutputs: vaeEncoder.outputNames,
	});

	sdTurboReady = true;
	modelProgressBar.style.width = "100%";
	modelProgressText.textContent = "100%";
	setStatus("Runtime pronto. Face detection + SD-Turbo bootstrap OK.");

	return sdSessions;
}

async function bootstrap() {
	try {
		setStatus("Caricamento face detection...");
		await initFaceDetection();

		if (supportsWebGPU()) {
			setStatus("Caricamento runtime SD-Turbo...");
			try {
				await initSdTurboSessions();
				setStatus("Runtime pronto. Face detection + SD-Turbo bootstrap OK.");
			} catch (sdErr) {
				console.warn(
					"[PATCH 2B] SD-Turbo non pronto, fallback placeholder:",
					sdErr,
				);
				sdTurboReady = false;
				modelProgressDetail.textContent +=
					"\n⚠️ Bootstrap SD‑Turbo fallito, uso fallback placeholder.";
				setStatus(
					"Face detection pronta. SD-Turbo non disponibile: uso fallback placeholder.",
				);
			}
		} else {
			sdTurboReady = false;
			setStatus(
				"Face detection pronta. WebGPU non disponibile: uso fallback placeholder.",
			);
		}
	} catch (err) {
		console.error(err);
		setStatus("Errore nel bootstrap iniziale.");
	}
}

// -------------------------
// detection
// -------------------------
async function detectSingleFaceFromCanvas(canvas) {
	const detection = await faceapi.detectSingleFace(canvas);
	if (!detection) return null;

	const { x, y, width, height } = detection.box;
	const padX = width * FACE_PADDING_RATIO;
	const padY = height * FACE_PADDING_RATIO;

	return {
		x: Math.floor(clamp(x - padX, 0, canvas.width)),
		y: Math.floor(clamp(y - padY, 0, canvas.height)),
		w: Math.floor(clamp(width + 2 * padX, 1, canvas.width)),
		h: Math.floor(clamp(height + 2 * padY, 1, canvas.height)),
	};
}

function cropRegionToSquare(
	sourceCanvas,
	regionBox,
	targetSize = FACE_INPUT_SIZE,
) {
	const cropCanvas = document.createElement("canvas");
	cropCanvas.width = targetSize;
	cropCanvas.height = targetSize;
	const ctx = cropCanvas.getContext("2d", { willReadFrequently: true });

	ctx.drawImage(
		sourceCanvas,
		regionBox.x,
		regionBox.y,
		regionBox.w,
		regionBox.h,
		0,
		0,
		targetSize,
		targetSize,
	);

	return cropCanvas;
}

function drawRegionOverlay(canvas, regionBox) {
	const overlay = document.createElement("canvas");
	overlay.width = canvas.width;
	overlay.height = canvas.height;
	const ctx = overlay.getContext("2d");

	ctx.drawImage(canvas, 0, 0);
	ctx.strokeStyle = "rgba(96,165,250,0.9)";
	ctx.lineWidth = Math.max(2, Math.floor(canvas.width / 300));
	ctx.strokeRect(regionBox.x, regionBox.y, regionBox.w, regionBox.h);

	return overlay;
}

// -------------------------
// fallback placeholder
// -------------------------
async function runPlaceholderTransform(cropCanvas, { prompt, strength, seed }) {
	console.log(
		"[PATCH 2B][fallback] Prompt:",
		prompt,
		"Strength:",
		strength,
		"Seed:",
		seed,
	);

	const out = document.createElement("canvas");
	out.width = cropCanvas.width;
	out.height = cropCanvas.height;
	const ctx = out.getContext("2d", { willReadFrequently: true });

	ctx.save();
	ctx.filter = `blur(${Math.max(2, Math.floor(strength * 8))}px) saturate(${0.85 + (1 - strength) * 0.2}) contrast(1.02)`;
	ctx.drawImage(cropCanvas, 0, 0);
	ctx.restore();

	let imageData = ctx.getImageData(0, 0, out.width, out.height);
	let data = imageData.data;

	let s = seed % 2147483647;
	if (s <= 0) s += 2147483646;
	const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;

	for (let i = 0; i < data.length; i += 4) {
		const n = (rnd() - 0.5) * 18 * strength;
		data[i] = clamp(data[i] + n + 4, 0, 255);
		data[i + 1] = clamp(data[i + 1] + n * 0.4, 0, 255);
		data[i + 2] = clamp(data[i + 2] - n * 0.8, 0, 255);
	}

	ctx.putImageData(imageData, 0, 0);

	const mirrorCanvas = document.createElement("canvas");
	mirrorCanvas.width = out.width;
	mirrorCanvas.height = out.height;
	const mctx = mirrorCanvas.getContext("2d");

	mctx.save();
	mctx.translate(out.width, 0);
	mctx.scale(-1, 1);
	mctx.drawImage(out, 0, 0);
	mctx.restore();

	ctx.save();
	ctx.globalAlpha = 0.12 + strength * 0.08;
	ctx.drawImage(
		mirrorCanvas,
		Math.floor(out.width * 0.08),
		0,
		Math.floor(out.width * 0.84),
		out.height,
		Math.floor(out.width * 0.08),
		0,
		Math.floor(out.width * 0.84),
		out.height,
	);
	ctx.restore();

	const warped = document.createElement("canvas");
	warped.width = out.width;
	warped.height = out.height;
	const wctx = warped.getContext("2d");

	const strips = 18;
	const stripH = Math.ceil(out.height / strips);

	for (let i = 0; i < strips; i++) {
		const sy = i * stripH;
		const sh = Math.min(stripH, out.height - sy);
		const dx = Math.floor((rnd() - 0.5) * 18 * strength);
		const dw = out.width + Math.floor((rnd() - 0.5) * 10 * strength);

		wctx.drawImage(out, 0, sy, out.width, sh, dx, sy, dw, sh);
	}

	wctx.save();
	const grad = wctx.createRadialGradient(
		warped.width / 2,
		warped.height / 2,
		warped.width * 0.2,
		warped.width / 2,
		warped.height / 2,
		warped.width * 0.62,
	);
	grad.addColorStop(0, "rgba(255,255,255,0)");
	grad.addColorStop(1, "rgba(0,0,0,0.07)");
	wctx.fillStyle = grad;
	wctx.fillRect(0, 0, warped.width, warped.height);
	wctx.restore();

	return warped;
}

// -------------------------
// SD-Turbo hook (bootstrap only for now)
// -------------------------
async function runSdTurboTransform(cropCanvas, { prompt, strength, seed }) {
	if (!sdTurboReady || !sdSessions) {
		throw new Error("SD-Turbo runtime non inizializzato.");
	}

	console.log(
		"[PATCH 2B][sd-turbo] Prompt:",
		prompt,
		"Strength:",
		strength,
		"Seed:",
		seed,
	);
	console.log(
		"[PATCH 2B][sd-turbo] Session bootstrap OK. Integrazione img2img reale da completare.",
	);

	// Per PATCH 2B manteniamo fallback visuale finché non colleghiamo tokenizer + unet + vae reale
	return runPlaceholderTransform(cropCanvas, { prompt, strength, seed });
}

// -------------------------
// compositing
// -------------------------
function createFeatherMask(w, h) {
	const mask = document.createElement("canvas");
	mask.width = w;
	mask.height = h;
	const ctx = mask.getContext("2d");

	const grad = ctx.createRadialGradient(
		w / 2,
		h / 2,
		Math.min(w, h) * 0.18,
		w / 2,
		h / 2,
		Math.min(w, h) * 0.52,
	);

	grad.addColorStop(0.0, "rgba(255,255,255,1)");
	grad.addColorStop(0.7, "rgba(255,255,255,0.90)");
	grad.addColorStop(1.0, "rgba(255,255,255,0)");

	ctx.fillStyle = grad;
	ctx.fillRect(0, 0, w, h);

	return mask;
}

function blendRegionBack(originalCanvas, generatedCanvas, regionBox) {
	const resultCanvas = document.createElement("canvas");
	resultCanvas.width = originalCanvas.width;
	resultCanvas.height = originalCanvas.height;
	const ctx = resultCanvas.getContext("2d", { willReadFrequently: true });

	ctx.drawImage(originalCanvas, 0, 0);

	const resized = document.createElement("canvas");
	resized.width = regionBox.w;
	resized.height = regionBox.h;
	const rctx = resized.getContext("2d", { willReadFrequently: true });
	rctx.drawImage(generatedCanvas, 0, 0, regionBox.w, regionBox.h);

	const mask = createFeatherMask(regionBox.w, regionBox.h);
	const masked = document.createElement("canvas");
	masked.width = regionBox.w;
	masked.height = regionBox.h;
	const mctx = masked.getContext("2d", { willReadFrequently: true });

	mctx.drawImage(resized, 0, 0);
	mctx.globalCompositeOperation = "destination-in";
	mctx.drawImage(mask, 0, 0);

	ctx.drawImage(masked, regionBox.x, regionBox.y);
	return resultCanvas;
}

// -------------------------
// UI wiring
// -------------------------
strengthSlider.addEventListener("input", () => {
	strengthValue.textContent = Number(strengthSlider.value).toFixed(2);
});

imageInput.addEventListener("change", async (e) => {
	const file = e.target.files?.[0];
	if (!file) return;

	try {
		setStatus("Caricamento immagine...");
		loadedImage = await loadImageFromFile(file);

		setCanvasSizeToImage(inputCanvas, inputCtx, loadedImage);
		copyCanvas(inputCanvas, outputCanvas, outputCtx);

		inputMeta.textContent = `Dimensioni: ${inputCanvas.width} × ${inputCanvas.height}`;
		outputMeta.textContent = sdTurboReady
			? "Runtime SD-Turbo bootstrap attivo."
			: "Fallback placeholder attivo.";

		runBtn.disabled = false;
		resetBtn.disabled = false;

		setStatus("Immagine caricata. Premi 'Anonymize'.");
	} catch (err) {
		console.error(err);
		setStatus("Errore nel caricamento dell'immagine.");
	}
});

resetBtn.addEventListener("click", () => {
	if (!loadedImage) return;

	setCanvasSizeToImage(inputCanvas, inputCtx, loadedImage);
	copyCanvas(inputCanvas, outputCanvas, outputCtx);
	outputMeta.textContent = "Output resettato.";
	setStatus("Reset completato.");
});

runBtn.addEventListener("click", async () => {
	if (!loadedImage) {
		setStatus("Carica prima un'immagine.");
		return;
	}

	try {
		runBtn.disabled = true;
		setStatus("Cerco la regione...");
		const regionBox = await detectSingleFaceFromCanvas(inputCanvas);

		if (!regionBox) {
			setStatus("Nessuna regione rilevata. Prova con un’immagine più chiara.");
			return;
		}

		const overlayCanvas = drawRegionOverlay(inputCanvas, regionBox);
		copyCanvas(overlayCanvas, inputCanvas, inputCtx);

		setStatus("Regione trovata. Eseguo crop...");
		const cropCanvas = cropRegionToSquare(
			outputCanvas.width ? outputCanvas : inputCanvas,
			regionBox,
			FACE_INPUT_SIZE,
		);

		const prompt = promptInput.value || "a realistic face";
		const strength = Number(strengthSlider.value);
		const seed = randomInt(1_000_000_000);

		setStatus(`Trasformazione in corso... (seed ${seed})`);

		const generated = sdTurboReady
			? await runSdTurboTransform(cropCanvas, { prompt, strength, seed })
			: await runPlaceholderTransform(cropCanvas, { prompt, strength, seed });

		console.log("[PATCH 2B] transform done");

		setStatus("Reinserimento nel canvas finale...");
		const resultCanvas = blendRegionBack(
			outputCanvas.width ? outputCanvas : inputCanvas,
			generated,
			regionBox,
		);

		outputCanvas.width = resultCanvas.width;
		outputCanvas.height = resultCanvas.height;
		outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
		outputCtx.drawImage(resultCanvas, 0, 0);

		console.log("[PATCH 2B] output draw done");

		outputMeta.textContent =
			`Box: x=${regionBox.x}, y=${regionBox.y}, w=${regionBox.w}, h=${regionBox.h}\n` +
			`prompt="${prompt}"\n` +
			`seed=${seed}\n` +
			`mode=${sdTurboReady ? "sd-turbo-bootstrap" : "placeholder-fallback"}`;

		setStatus(
			sdTurboReady
				? "Completato. Runtime CDN + SD-Turbo bootstrap attivo."
				: "Completato. Fallback placeholder attivo.",
		);
	} catch (err) {
		console.error(err);
		setStatus("Errore durante la trasformazione. Controlla la console.");
	} finally {
		runBtn.disabled = false;
	}
});

// bootstrap
bootstrap();
