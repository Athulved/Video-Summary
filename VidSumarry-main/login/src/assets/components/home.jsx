// src/pages/Home.jsx
import React, { useEffect, useState } from "react";
import { auth, storage, db } from "./firebase";
import { signOut } from "firebase/auth";
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { collection, addDoc } from "firebase/firestore";
import { useNavigate } from "react-router-dom";

const Home = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  // Check if the user is authenticated
  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (!user) {
        navigate("/login"); // Redirect to login if not authenticated
      }
    });

    return () => unsubscribe(); // Cleanup subscription
  }, [navigate]);

  // Handle logout
  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigate("/login"); // Redirect to login after logout
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  // Handle video upload
  const handleUpload = async () => {
    if (!file) {
      alert("Please select a video file to upload.");
      return;
    }

    setUploading(true);
    try {
      const user = auth.currentUser; // Get the currently logged-in user
      if (!user) {
        throw new Error("User not authenticated.");
      }

      // Upload video to Firebase Storage
      const storageRef = ref(storage, `videos/${user.uid}/${file.name}`);
      await uploadBytes(storageRef, file);
      const downloadURL = await getDownloadURL(storageRef);

      console.log("Video uploaded to Storage. Download URL:", downloadURL);

      // Store video metadata in Firestore
      const docRef = await addDoc(collection(db, "videos"), {
        fileName: file.name,
        fileURL: downloadURL,
        timestamp: new Date(),
        userId: user.uid, // Associate the video with the user's UID
      });

      console.log("Video metadata added to Firestore with ID:", docRef.id);

      alert("Video uploaded successfully!");
      setFile(null); // Clear the file input
    } catch (error) {
      console.error("Error uploading video:", error);
      alert("Failed to upload video.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1>Welcome to the Home Page!</h1>
      <p>You are now logged in.</p>

      {/* Video Upload Section */}
      <div style={styles.uploadSection}>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files[0])}
          style={styles.fileInput}
        />
        <button onClick={handleUpload} disabled={uploading} style={styles.uploadButton}>
          {uploading ? "Uploading..." : "Upload Video"}
        </button>
      </div>

      {/* Button to Navigate to Video History */}
      <button onClick={() => navigate("/video-history")} style={styles.historyButton}>
        View Video History
      </button>

      {/* Logout Button */}
      <button onClick={handleLogout} style={styles.button}>
        Logout
      </button>
    </div>
  );
};

// Styles
const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    minHeight: "100vh",
    backgroundColor: "black", // Black background
    color: "white", // White text for readability
    textAlign: "center",
  },
  uploadSection: {
    margin: "20px 0",
  },
  fileInput: {
    marginBottom: "10px",
  },
  uploadButton: {
    padding: "10px 20px",
    backgroundColor: "#007bff",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontSize: "16px",
  },
  historyButton: {
    padding: "10px 20px",
    backgroundColor: "#28a745",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontSize: "16px",
    margin: "10px 0",
  },
  button: {
    padding: "10px 20px",
    backgroundColor: "#ff4444",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontSize: "16px",
    marginTop: "20px",
  },
};

export default Home;