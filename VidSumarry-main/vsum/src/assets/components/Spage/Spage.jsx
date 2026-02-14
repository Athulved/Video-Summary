import React from 'react';
import Navbar from './Navbar';
import VideoUpload from './VideoUpload';
import './Spage.css';
import { useNavigate } from "react-router-dom";
import { auth } from "../firebase";
import { useEffect } from "react";


function Spage() {
  const navigate = useNavigate();
  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (!user) {
        navigate("/login"); // Redirect to login if not authenticated
      } else {
        console.log("User is authenticated:", user); // Debugging
      }
    });
  
    return () => unsubscribe(); // Cleanup subscription
  }, [navigate]);
  return (
    <div className="Spage">
      {/* <div className="blur-overlay">
  
      </div> */}
      <svg preserveAspectRatio="xMidYMid slice" viewBox="10 10 80 80">
        {/* SVG paths here */}
      </svg>
      <Navbar />
      <div className="scrollable-content">
       
        <VideoUpload />
        {/* <Summary /> */}
      </div>
    </div>
  );
}

export default Spage;
