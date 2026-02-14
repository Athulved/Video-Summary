// src/pages/VideoHistory.jsx
import React, { useEffect, useState } from "react";
import { db, auth } from "./firebase";
import { collection, query, where, getDocs } from "firebase/firestore";
import { useNavigate } from "react-router-dom";

const VideoHistory = () => {
  const [videos, setVideos] = useState([]);
  const navigate = useNavigate();

  // Fetch video history for the logged-in user
  useEffect(() => {
    const fetchVideos = async () => {
      const user = auth.currentUser; // Get the currently logged-in user
      if (!user) {
        navigate("/login"); // Redirect to login if not authenticated
        return;
      }

      const videosCollection = collection(db, "videos");
      const q = query(videosCollection, where("userId", "==", user.uid)); // Filter by user UID
      const videosSnapshot = await getDocs(q);
      const videosList = videosSnapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));

      console.log("Fetched videos:", videosList); // Log fetched videos

      setVideos(videosList);
    };

    fetchVideos();
  }, [navigate]);

  return (
    <div style={styles.container}>
      <h1>Video History</h1>
      <button onClick={() => navigate("/")} style={styles.backButton}>
        Back to Home
      </button>
      {videos.length === 0 ? (
        <p>No videos uploaded yet.</p>
      ) : (
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.tableHeader}>File Name</th>
              <th style={styles.tableHeader}>Uploaded At</th>
            </tr>
          </thead>
          <tbody>
            {videos.map((video) => (
              <tr key={video.id} style={styles.tableRow}>
                <td style={styles.tableCell}>{video.fileName}</td>
                <td style={styles.tableCell}>
                  {new Date(video.timestamp?.toDate()).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

// Styles
const styles = {
  container: {
    padding: "20px",
    textAlign: "center",
    backgroundColor: "black", // Black background
    color: "white", // White text for readability
    minHeight: "100vh",
  },
  backButton: {
    padding: "10px 20px",
    backgroundColor: "#007bff",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontSize: "16px",
    marginBottom: "20px",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    margin: "20px 0",
  },
  tableHeader: {
    backgroundColor: "#007bff",
    color: "white",
    padding: "10px",
    border: "1px solid #ddd",
  },
  tableRow: {
    border: "1px solid #ddd",
  },
  tableCell: {
    padding: "10px",
    border: "1px solid #ddd",
  },
};

export default VideoHistory;