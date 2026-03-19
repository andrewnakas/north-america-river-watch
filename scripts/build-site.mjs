import { mkdir, cp, rm, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const srcDir = path.join(root, 'src');
const generatedDir = path.join(root, 'generated');
const distDir = path.join(root, 'dist');

async function exists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

await rm(distDir, { recursive: true, force: true });
await mkdir(distDir, { recursive: true });
await cp(srcDir, distDir, { recursive: true });

if (await exists(generatedDir)) {
  await cp(generatedDir, path.join(distDir, 'data'), { recursive: true });
  console.log('Built site into dist/ with local generated data');
} else {
  console.log('Built site into dist/ without local generated data (runtime fallback enabled)');
}
