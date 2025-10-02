# 📱 How to Use on Your Phone

## ✅ **Server is Already Running!**

Your demo server is active and ready for mobile access.

---

## 🎯 **3 SIMPLE STEPS:**

### **STEP 1: Make Sure Both Devices Are on Same WiFi**

Your **computer** and **phone** must be connected to the **same WiFi network**.

---

### **STEP 2: Open This URL on Your Phone**

Open your phone's browser (Safari, Chrome, etc.) and go to:

```
http://172.17.75.158:8080
```

**Or scan this with your camera:**
(If you have a QR code app, scan it to open directly)

---

### **STEP 3: Use the App!**

1. **Click "Start Camera"** button
2. **Allow camera permissions** when prompted
3. **Point camera at a plant leaf**
4. **Click "Capture & Analyze"**
5. **See the disease prediction!**

---

## 📝 **Detailed Instructions:**

### **On Your Phone:**

**1. Connect to WiFi**
- Make sure you're on the same WiFi as your computer
- Both devices must be on the same network

**2. Open Browser**
- Open Safari, Chrome, or any browser
- Type in the address bar:
  ```
  172.17.75.158:8080
  ```
  (Note: No "www" needed, include the port :8080)

**3. Grant Camera Permission**
- When prompted, tap **"Allow"** for camera access
- If you accidentally denied it:
  - Go to Settings → Safari → Camera
  - Set to "Allow"

**4. Use the App**
- Tap **"📷 Start Camera"**
- Point at a plant leaf (or image on screen)
- Tap **"📸 Capture & Analyze"**
- Wait 1-2 seconds for prediction
- See results with confidence scores!

---

## 🔧 **Troubleshooting:**

### **❌ Can't Connect?**

**Check 1: Same WiFi?**
```
Make sure both phone and computer are on the SAME WiFi network
```

**Check 2: Server Running?**
```
On your computer, check if you see:
"Running on http://172.17.75.158:8080"
```

**Check 3: Firewall?**
```
macOS Firewall might block connections
Go to: System Preferences → Security & Privacy → Firewall
Temporarily disable or allow Python
```

**Check 4: Try Different Browser**
```
If Safari doesn't work, try Chrome or Firefox on your phone
```

---

### **❌ Camera Not Working?**

**Solution 1: Grant Permissions**
```
Settings → Safari → Camera → Allow
or
Settings → Chrome → Camera → Allow
```

**Solution 2: Use HTTPS**
```
Some browsers require HTTPS for camera
Try: https://172.17.75.158:8080
(You'll get a security warning - click "Proceed Anyway")
```

**Solution 3: Different Browser**
```
Try Chrome, Firefox, or Edge on your phone
```

---

## 💡 **Tips for Best Results:**

✅ **Use good lighting** - Take photos in well-lit areas  
✅ **Get close** - Fill the frame with the leaf  
✅ **Focus clearly** - Make sure the image isn't blurry  
✅ **Try different angles** - Sometimes a different angle works better  
✅ **Works with images** - Can point at leaf images on another screen  

---

## 🌐 **Alternative Access Methods:**

### **Method 1: Direct IP (What we're using)**
```
http://172.17.75.158:8080
```

### **Method 2: Local Only**
```
http://localhost:8080
(Only works on your computer)
```

### **Method 3: Find IP Manually**

On your computer, run:
```bash
ifconfig | grep "inet "
```

Look for the IP address (usually 192.168.x.x or 172.x.x.x)

---

## 📊 **What You Can Test:**

Your AI can detect **45 crop diseases**:

**Popular Crops:**
- 🍅 Tomato (10 diseases)
- 🥔 Potato (3 diseases)  
- 🌽 Corn (4 diseases)
- 🍎 Apple (4 diseases)
- 🍇 Grape (4 diseases)
- 🌶️ Pepper (2 diseases)
- 🌾 Rice (4 diseases)
- 🌱 Cotton (4 diseases)

**Test Ideas:**
- Take photos of real plant leaves
- Point at plant images on another screen
- Try different crops
- Compare confidence scores

---

## 🎬 **Quick Start Checklist:**

- [ ] Computer and phone on **same WiFi**
- [ ] Demo server **running** on computer
- [ ] Open **http://172.17.75.158:8080** on phone
- [ ] **Allow camera** permissions
- [ ] Click **"Start Camera"**
- [ ] **Capture** and get prediction!

---

## 🛑 **If You Need to Restart Server:**

**Stop the server:**
```bash
# Press Ctrl+C in the terminal where server is running
```

**Start again:**
```bash
cd /Users/jayadharunr/Crop_disease_Prediction
python3 src/demo_server.py
```

---

## 📞 **Quick Reference:**

**Your Computer's IP:**
```
172.17.75.158
```

**URL for Phone:**
```
http://172.17.75.158:8080
```

**Server Port:**
```
8080
```

**Server Status:**
```
✅ Running and ready for connections!
```

---

## 🎉 **Ready to Test!**

**On your phone, open:**
```
http://172.17.75.158:8080
```

**Then:**
1. Tap "Start Camera"
2. Point at leaf
3. Tap "Capture & Analyze"
4. See the magic! 🌱✨

---

**Have fun testing your AI crop disease detector!** 📱🌿🔬

