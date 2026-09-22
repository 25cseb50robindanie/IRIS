const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');

function createWindow() {
  const iconPath = path.join(__dirname, 'icon.png');
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    icon: iconPath,
    webPreferences: {
      // Security: Electron defaults, never override these
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  const distHtml = path.join(__dirname, '../dist/index.html');
  const fs = require('fs');

  if (fs.existsSync(distHtml)) {
    win.loadFile(distHtml);
  } else {
    win.loadURL('http://127.0.0.1:5173');
  }
}

// IPC: native folder/file picker — exposed via preload, never raw fs access
ipcMain.handle('dialog:openFile', async (_event, options) => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      { name: 'Satellite Imagery', extensions: ['tif', 'tiff', 'jp2'] },
      { name: 'All Files', extensions: ['*'] },
    ],
    ...options,
  });
  if (result.canceled) return null;
  return result.filePaths[0];
});

ipcMain.handle('dialog:openFolder', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  });
  if (result.canceled) return null;
  return result.filePaths[0];
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  app.quit();
});
