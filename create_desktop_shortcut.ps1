$desktop = [Environment]::GetFolderPath('Desktop')
$ws = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path $desktop "IRIS.lnk"

# Remove existing shortcut to force Windows shell recreation
if (Test-Path $shortcutPath) {
    Remove-Item -Force $shortcutPath
}

$s = $ws.CreateShortcut($shortcutPath)
$s.TargetPath = "c:\Danie\Projects\IRIS\Launch_IRIS.bat"
$s.WorkingDirectory = "c:\Danie\Projects\IRIS"
$s.IconLocation = "c:\Danie\Projects\IRIS\iris_app.ico,0"
$s.Description = "IRIS - Satellite Intelligence System (Offline Desktop)"
$s.Save()

# Notify Windows Shell to immediately flush and reload icon cache
$code = @'
using System;
using System.Runtime.InteropServices;
public class ShellNotifier {
    [DllImport("Shell32.dll")]
    public static extern void SHChangeNotify(int wEventId, int uFlags, IntPtr dwItem1, IntPtr dwItem2);
}
'@
Add-Type -TypeDefinition $code -ErrorAction SilentlyContinue
[ShellNotifier]::SHChangeNotify(0x08000000, 0x0000, [IntPtr]::Zero, [IntPtr]::Zero)

Write-Output "Successfully updated desktop shortcut at: $shortcutPath"
