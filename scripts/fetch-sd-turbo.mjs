import { mkdir, access, writeFile } from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const DEST_BASE = path.join(ROOT, "models", "sd-turbo");

const FILES = [
	{
		name: "text_encoder/model.onnx",
		urls: [
			"https://huggingface.co/microsoft/sd-turbo-webnn/resolve/main/text_encoder/model.onnx",
			"https://huggingface.co/webnn/sd-turbo-webnn/resolve/main/text_encoder/model.onnx",
		],
	},
	{
		name: "unet/model.onnx",
		urls: [
			"https://huggingface.co/microsoft/sd-turbo-webnn/resolve/main/unet/model.onnx",
			"https://huggingface.co/webnn/sd-turbo-webnn/resolve/main/unet/model.onnx",
		],
	},
	{
		name: "vae_decoder/model.onnx",
		urls: [
			"https://huggingface.co/microsoft/sd-turbo-webnn/resolve/main/vae_decoder/model.onnx",
			"https://huggingface.co/webnn/sd-turbo-webnn/resolve/main/vae_decoder/model.onnx",
		],
	},
	{
		name: "vae_encoder/model.onnx",
		urls: [
			"https://huggingface.co/eyaler/sd-turbo-webnn/resolve/main/vae_encoder/model.onnx",
		],
	},
];

async function exists(filePath) {
	try {
		await access(filePath);
		return true;
	} catch {
		return false;
	}
}

async function downloadWithFallback(urls, destPath) {
	let lastError = null;

	for (const url of urls) {
		try {
			console.log(`[fetch-sd-turbo] Downloading: ${url}`);
			const res = await fetch(url);

			if (!res.ok) {
				throw new Error(`HTTP ${res.status} ${res.statusText}`);
			}

			const arrayBuffer = await res.arrayBuffer();
			const data = Buffer.from(arrayBuffer);

			await mkdir(path.dirname(destPath), { recursive: true });
			await writeFile(destPath, data);

			console.log(`[fetch-sd-turbo] Saved: ${destPath}`);
			return;
		} catch (err) {
			console.warn(`[fetch-sd-turbo] Failed: ${url} -> ${err.message}`);
			lastError = err;
		}
	}

	throw lastError ?? new Error(`All download attempts failed for ${destPath}`);
}

async function main() {
	await mkdir(DEST_BASE, { recursive: true });

	for (const file of FILES) {
		const destPath = path.join(DEST_BASE, file.name);

		if (await exists(destPath)) {
			console.log(`[fetch-sd-turbo] Skip existing: ${destPath}`);
			continue;
		}

		await downloadWithFallback(file.urls, destPath);
	}

	console.log("[fetch-sd-turbo] SD-Turbo model fetch complete.");
}

main().catch((err) => {
	console.error("[fetch-sd-turbo] fatal error:", err);
	process.exit(1);
});
