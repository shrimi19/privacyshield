# PrivacyShield - Run Steps

These steps are generic for anyone who clones the repository.

## 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd privacyshield
```

## 2. Create a virtual environment

### Windows (PowerShell)
```powershell
python -m venv .venv
```

### macOS / Linux
```bash
python3 -m venv .venv
```

## 3. Activate the virtual environment

### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the app

```bash
python app.py
```

## 6. Open the website

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

## 7. Optional health check from terminal

### Windows (PowerShell)
```powershell
try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000).StatusCode } catch { $_.Exception.Message }
```

### macOS / Linux
```bash
curl -I http://127.0.0.1:5000
```

## Troubleshooting

### PowerShell blocks activation script

Run this once in the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Port already in use

If `5000` is busy, stop the existing process using that port, then run `python app.py` again.
