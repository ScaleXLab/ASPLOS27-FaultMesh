# EasyConnect VPN

Campus VPN client. `EasyConnectInstaller.exe` is the Windows installer, and `EasyConnect_7_6_7_4.dmg` is the macOS installer. `Easyconnect VPN Remote Access Guide of Easyconnect VPN_522.pdf` explains how to install and use the VPN.

Download both `EasyConnectInstaller.exe` and `EasyConnect_7_6_7_4.dmg` onto your local machine and install them following the instructions in the pdf.

Please check the VPN account and VPN password, the server IP address and password, and the sudo password in the HotCRP comments.

## Download the macOS installer

`EasyConnect_7_6_7_4.dmg` is larger than GitHub's file size limit, so it is stored with Git LFS. A normal `git clone` or `git pull` only fetches a small pointer file, not the installer itself.

From the repository root:

```bash
git lfs install
git lfs pull --include="vpn/*.dmg"
```

When this finishes, `vpn/EasyConnect_7_6_7_4.dmg` should be about 126MB. If Git LFS is not installed, use `sudo apt install git-lfs` on Ubuntu.
