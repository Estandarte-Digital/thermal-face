sc stop "Brick Daemon"
del /f C:\ProgramData\Tinkerforge\Brickd\brickd.log
sc start "Brick Daemon"