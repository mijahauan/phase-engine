#!/usr/bin/env bash

# Exit on error
set -e

echo "Installing Phase Engine to /opt/phase-engine..."

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./install.sh)"
  exit 1
fi

# Create dedicated system user
if ! id -u phase-engine >/dev/null 2>&1; then
    echo "Creating phase-engine user..."
    useradd -r -s /bin/false phase-engine
fi

# Setup directories
echo "Setting up directories..."
mkdir -p /opt/phase-engine
mkdir -p /etc/phase-engine

# Copy application files
echo "Copying application files..."
cp -r src scripts systemd pyproject.toml README.md /opt/phase-engine/

# Copy config if it doesn't exist
if [ ! -f /etc/phase-engine/config.toml ]; then
    echo "Copying default config to /etc/phase-engine/config.toml..."
    cp config/phase-engine.toml.template /etc/phase-engine/config.toml
else
    echo "Config file already exists at /etc/phase-engine/config.toml, skipping."
fi

# Setup Python venv
echo "Setting up Python virtual environment..."
cd /opt/phase-engine
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -e .

# Set permissions
echo "Setting permissions..."
chown -R phase-engine:phase-engine /opt/phase-engine
chown -R phase-engine:phase-engine /etc/phase-engine

# Install systemd service
echo "Installing systemd service..."
cp /opt/phase-engine/systemd/phase-engine.service /etc/systemd/system/
systemctl daemon-reload

echo ""
echo "Installation complete!"
echo "Next steps:"
echo "1. Edit configuration: sudo nano /etc/phase-engine/config.toml"
echo "2. Start the daemon:   sudo systemctl start phase-engine"
echo "3. Enable on boot:     sudo systemctl enable phase-engine"
echo "4. View logs:          sudo journalctl -u phase-engine -f"
