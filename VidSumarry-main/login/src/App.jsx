// src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./assets/components/login";
import Register from "./assets/components/register";
import Home from "./assets/components/home";
import VideoHistory from "./assets/components/VideoHistory";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/" element={<Home />} />
        <Route path="/video-history" element={<VideoHistory />} />
        <Route path="*" element={<Navigate to="/" />} /> {/* Redirect unknown paths to Home */}
      </Routes>
    </Router>
  );
};

export default App;