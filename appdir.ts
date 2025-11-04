import path from "path";
import fs from 'fs/promises';

export async function ensureDirExists(dir: string) {
    try {
        await fs.mkdir(dir, { recursive: true });
    } catch (e) {
        console.error(`Error creating directory ${dir}:`, e);
    }
}

export const appdir = () => process.env.APPDATA || (process.platform == 'darwin' ? process.env.HOME + '/Library/Preferences' : process.env.HOME + "/.local/share")
export const APP_NAME = "unblink-engine";
export const RUNTIME_DIR = path.join(appdir(), APP_NAME);
export const FILES_DIR = (tenant_id: string) => path.join(RUNTIME_DIR, tenant_id, 'files');
export const FRAMES_DIR = (tenant_id: string) => path.join(FILES_DIR(tenant_id), 'frames');

// Create directories if they don't exist
await ensureDirExists(RUNTIME_DIR);