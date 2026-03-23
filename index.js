/* PATCH 1 - Face anonymizer scaffold
 *
 * Questa patch:
 * - carica un'immagine
 * - rileva un volto
 * - ritaglia il volto con padding
 * - applica una "anonymization" placeholder (blur + warp + noise + tone shift)
 * - re-blenda il volto nell'immagine originale
 *
 * PATCH 2:
 * - sostituire runImg2ImgOnCrop() con la pipeline reale SD-Turbo img2img
 */

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

const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });
const outputCtx = outputCanvas.getContext("2d", { willReadFrequently: true });

const FACE_MODEL_URL = "/models/face-api";
const FACE_INPUT_SIZE = 512;
const FACE_PADDING_RATIO = 0.28;

let loadedImage = null;
let originalImageBitmap = null;
let lastFaceBox = null;

// -------------------------
// UI helpers
// -------------------------
function setStatus(msg) {
	statusEl.textContent = msg;
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

function clamp(value, min, max) {
	return Math.max(min, Math.min(max, value));
}

function randomInt(max) {
	return Math.floor(Math.random() * max);
}

// -------------------------
// Initialization
// -------------------------
async function init() {
	try {
		setStatus("Caricamento modelli face detection...");
		// SSD MobileNet V1
		await faceapi.nets.ssdMobilenetv1.loadFromUri(
			`${FACE_MODEL_URL}/ssd_mobilenetv1`,
		);
		setStatus("Modelli caricati. Ora puoi caricare un'immagine.");
	} catch (err) {
		console.error(err);
		setStatus(
			"Errore nel caricamento dei modelli face-api. Controlla la cartella /models/face-api/ssd_mobilenetv1.",
		);
	}
}

strengthSlider.addEventListener("input", () => {
	strengthValue.textContent = Number(strengthSlider.value).toFixed(2);
});

// -------------------------
// Image loading
// -------------------------
function loadImageFromFile(file) {
	return new Promise((resolve, reject) => {
		const img = new Image();
		img.onload = () => resolve(img);
		img.onerror = reject;
		img.src = URL.createObjectURL(file);
	});
}

imageInput.addEventListener("change", async (e) => {
	const file = e.target.files?.[0];
	if (!file) return;

	try {
		setStatus("Caricamento immagine...");
		loadedImage = await loadImageFromFile(file);
		originalImageBitmap = loadedImage;

		setCanvasSizeToImage(inputCanvas, inputCtx, loadedImage);
		copyCanvas(inputCanvas, outputCanvas, outputCtx);

		inputMeta.textContent = `Dimensioni: ${inputCanvas.width} × ${inputCanvas.height}`;
		outputMeta.textContent = "Output pronto per la generazione.";
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

// -------------------------
// Face detection and crop
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

function cropFaceToSquare(sourceCanvas, faceBox, targetSize = FACE_INPUT_SIZE) {
	const cropCanvas = document.createElement("canvas");
	cropCanvas.width = targetSize;
	cropCanvas.height = targetSize;

	const ctx = cropCanvas.getContext("2d", { willReadFrequently: true });

	ctx.drawImage(
		sourceCanvas,
		faceBox.x,
		faceBox.y,
		faceBox.w,
		faceBox.h,
		0,
		0,
		targetSize,
		targetSize,
	);

	return cropCanvas;
}

// -------------------------
// Placeholder anonymization
// -------------------------
/**
 * PATCH 2:
 * sostituisci il contenuto di questa funzione con la vera pipeline SD-Turbo img2img.
 *
 * Firma consigliata per il futuro:
 *   runImg2ImgOnCrop(cropCanvas, {
 *      prompt,
 *      strength,
 *      seed
 *   }) => Promise<HTMLCanvasElement>
 */
async function runImg2ImgOnCrop(cropCanvas, { prompt, strength, seed }) {
	// Per ora facciamo una trasformazione "placeholder" ma convincente:
	// - blur selettivo
	// - lieve deformazione affine a blocchi
	// - shift cromatico
	// - noise leggero
	// - overlay speculare parziale
	//
	// L'idea è simulare un volto "diverso" pur mantenendo posizione/illuminazione generale.
	// Nella PATCH 2 qui dentro andrà il vero ramo ONNX img2img.

	console.log(
		"[PATCH 1] Prompt:",
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

	// base
	ctx.save();
	ctx.filter = `blur(${Math.max(2, Math.floor(strength * 8))}px) saturate(${0.85 + (1 - strength) * 0.2}) contrast(1.02)`;
	ctx.drawImage(cropCanvas, 0, 0);
	ctx.restore();

	// copy data for pixel ops
	let imageData = ctx.getImageData(0, 0, out.width, out.height);
	let data = imageData.data;

	// simple pseudo random from seed
	let s = seed % 2147483647;
	if (s <= 0) s += 2147483646;
	const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;

	// subtle tone/noise shift
	for (let i = 0; i < data.length; i += 4) {
		const n = (rnd() - 0.5) * 18 * strength;
		data[i] = clamp(data[i] + n + 4, 0, 255); // R
		data[i + 1] = clamp(data[i + 1] + n * 0.4, 0, 255); // G
		data[i + 2] = clamp(data[i + 2] - n * 0.8, 0, 255); // B
	}
	ctx.putImageData(imageData, 0, 0);

	// mirror overlay in central region for identity disruption
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

	// warp strips
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

	// soft vignette to hide seams / make blend easier
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
// Blending back
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

function blendFaceBack(originalCanvas, generatedFaceCanvas, faceBox) {
	const resultCanvas = document.createElement("canvas");
	resultCanvas.width = originalCanvas.width;
	resultCanvas.height = originalCanvas.height;
	const ctx = resultCanvas.getContext("2d", { willReadFrequently: true });

	// original
	ctx.drawImage(originalCanvas, 0, 0);

	// resize generated face to bbox
	const resizedFace = document.createElement("canvas");
	resizedFace.width = faceBox.w;
	resizedFace.height = faceBox.h;
	const rctx = resizedFace.getContext("2d", { willReadFrequently: true });
	rctx.drawImage(generatedFaceCanvas, 0, 0, faceBox.w, faceBox.h);

	// feather mask
	const mask = createFeatherMask(faceBox.w, faceBox.h);

	const maskedFace = document.createElement("canvas");
	maskedFace.width = faceBox.w;
	maskedFace.height = faceBox.h;
	const mctx = maskedFace.getContext("2d", { willReadFrequently: true });

	mctx.drawImage(resizedFace, 0, 0);
	mctx.globalCompositeOperation = "destination-in";
	mctx.drawImage(mask, 0, 0);

	// draw final
	ctx.drawImage(maskedFace, faceBox.x, faceBox.y);

	return resultCanvas;
}

function drawFaceBoxOverlay(canvas, faceBox) {
	const overlay = document.createElement("canvas");
	overlay.width = canvas.width;
	overlay.height = canvas.height;
	const ctx = overlay.getContext("2d");
	ctx.drawImage(canvas, 0, 0);

	ctx.strokeStyle = "rgba(96,165,250,0.9)";
	ctx.lineWidth = Math.max(2, Math.floor(canvas.width / 300));
	ctx.strokeRect(faceBox.x, faceBox.y, faceBox.w, faceBox.h);

	return overlay;
}

// -------------------------
// Main pipeline
// -------------------------
runBtn.addEventListener("click", async () => {
	if (!loadedImage) {
		setStatus("Carica prima un'immagine.");
		return;
	}

	try {
		runBtn.disabled = true;
		setStatus("Cerco il volto...");

		const faceBox = await detectSingleFaceFromCanvas(inputCanvas);

		if (!faceBox) {
			setStatus(
				"Nessun volto rilevato. Prova con una foto più frontale o con il volto più grande.",
			);
			runBtn.disabled = false;
			return;
		}

		lastFaceBox = faceBox;

		// disegna box di debug sull'input (solo visivo)
		const overlayCanvas = drawFaceBoxOverlay(inputCanvas, faceBox);
		copyCanvas(overlayCanvas, inputCanvas, inputCtx);

		setStatus("Volto trovato. Eseguo crop...");
		const cropCanvas = cropFaceToSquare(
			outputCanvas.width
				? outputCanvas
				: originalImageBitmap instanceof HTMLImageElement
					? inputCanvas
					: inputCanvas,
			faceBox,
			FACE_INPUT_SIZE,
		);

		const prompt = promptInput.value || "a realistic face";
		const strength = Number(strengthSlider.value);
		const seed = randomInt(1_000_000_000);

		setStatus(`Anonimizzazione in corso... (seed ${seed})`);

		const generatedFace = await runImg2ImgOnCrop(cropCanvas, {
			prompt,
			strength,
			seed,
		});
		console.log("Generated faces!");

		setStatus("Reinserisco il volto nell'immagine originale...");
		const resultCanvas = blendFaceBack(
			outputCanvas.width ? outputCanvas : inputCanvas,
			generatedFace,
			faceBox,
		);

		outputCanvas.width = resultCanvas.width;
		outputCanvas.height = resultCanvas.height;
		outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
		outputCtx.drawImage(resultCanvas, 0, 0);
		console.log("Inserted face!");

		outputMeta.textContent = `Face box: x=${faceBox.x}, y=${faceBox.y}, w=${faceBox.w}, h=${faceBox.h} | prompt="${prompt}" | seed=${seed}`;
		setStatus(
			"Completato. PATCH 1 attiva: pipeline pronta per sostituire il placeholder con SD‑Turbo img2img.",
		);
	} catch (err) {
		console.error(err);
		setStatus("Errore durante l'anonimizzazione. Controlla la console.");
	} finally {
		runBtn.disabled = false;
	}
});

// bootstrap
init();
