"""IRIS Native Windows Dialog API — opens the real OS folder/file picker on Windows even in browser mode."""

import logging
import subprocess
import sys
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("iris.api.dialog")
router = APIRouter(prefix="/api/dialog", tags=["dialog"])


class DialogResult(BaseModel):
    path: Optional[str] = None


def _show_windows_folder_dialog() -> Optional[str]:
    cmd = """
    Add-Type -AssemblyName System.Windows.Forms
    $d = New-Object System.Windows.Forms.FolderBrowserDialog
    $d.Description = 'Select Satellite Imagery Folder (.SAFE or Landsat)'
    $d.ShowNewFolderButton = $false
    $f = New-Object System.Windows.Forms.Form
    $f.TopMost = $true
    if ($d.ShowDialog($f) -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::WriteLine($d.SelectedPath)
    }
    """
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
        out = res.stdout.strip()
        return out if out else None
    except Exception as exc:
        logger.warning("Could not open Windows folder dialog: %s", exc)
        return None


def _show_windows_file_dialog() -> Optional[str]:
    cmd = """
    Add-Type -AssemblyName System.Windows.Forms
    $d = New-Object System.Windows.Forms.OpenFileDialog
    $d.Title = 'Select Satellite Imagery File (GeoTIFF / JP2)'
    $d.Filter = 'Satellite Imagery (*.tif;*.tiff;*.jp2)|*.tif;*.tiff;*.jp2|All Files (*.*)|*.*'
    $f = New-Object System.Windows.Forms.Form
    $f.TopMost = $true
    if ($d.ShowDialog($f) -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::WriteLine($d.FileName)
    }
    """
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
        out = res.stdout.strip()
        return out if out else None
    except Exception as exc:
        logger.warning("Could not open Windows file dialog: %s", exc)
        return None


@router.post("/pick-folder", response_model=DialogResult)
def pick_folder() -> DialogResult:
    if sys.platform != "win32":
        return DialogResult(path=None)
    return DialogResult(path=_show_windows_folder_dialog())


@router.post("/pick-file", response_model=DialogResult)
def pick_file() -> DialogResult:
    if sys.platform != "win32":
        return DialogResult(path=None)
    return DialogResult(path=_show_windows_file_dialog())
