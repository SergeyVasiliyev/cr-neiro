# Turns the monitor(s) off now; the computer keeps working. Move the mouse or press a key to wake them.
Add-Type -Namespace Win32 -Name Monitor -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool PostMessage(int hWnd, int msg, int wParam, int lParam);
'@
Start-Sleep -Seconds 1  # let the click that started this settle, or it wakes the screen right away
[Win32.Monitor]::PostMessage(0xFFFF, 0x0112, 0xF170, 2) | Out-Null  # broadcast WM_SYSCOMMAND / SC_MONITORPOWER off
