const { contextBridge, ipcRenderer } = require('electron');

// Only expose specific, named functions — never raw ipcRenderer
contextBridge.exposeInMainWorld('iris', {
  openFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
  openFolder: () => ipcRenderer.invoke('dialog:openFolder'),
});
