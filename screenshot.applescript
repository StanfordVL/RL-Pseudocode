-- Enable screenshot with filename prompt, and then add to Automator Service --
-- 1. Open Automator.  2. Make a new Service. 3. Make sure it receives 'no input' at all programs. 4. Select Run Apple Script from Library menu and type in your code. --
-- Now go to System Preferences > Keyboard > Shortcuts. Select Services from the sidebar and find your service. Add a shortcut by double clicking (none). --
-- set f to (choose file name default location (path to desktop) Â --

set f to (choose file name Â
    default name (do shell script "date +'Screen_%Y%m%d_%H%M%S.png'"))'s POSIX path
do shell script "screencapture -i " & f's quoted form
