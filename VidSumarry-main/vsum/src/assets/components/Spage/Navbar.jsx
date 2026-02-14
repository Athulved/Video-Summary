import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Navbar.css';
import { auth } from "../firebase";
import { signOut } from "firebase/auth";

function Navbar() {
  const move=useNavigate();
  const [open, setOpen] = useState(false);
  const toggleMenu = () => setOpen(!open);
  const handleLogout = async () => {
    try {
      await signOut(auth);
      move("/"); // Redirect to login after logout
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };
  return (
    <nav className="navbar">
      <div className="keyboard">
        <span className="key">B</span>
        <span className="key">R</span>
        <span className="key">E</span>
        <span className="key">E</span>
        <span className="key">Z</span>
        <span className="key">I</span>
        <span className="key">P</span>
        {/* <span className="key">R</span>
        <span className="key">e</span>
        <span className="key">c</span>
        <span className="key">a</span>
        <span className="key">p</span> */}
      </div>

      <div className="nav-container">
        <button onClick={()=>move('/Mainpage')}className="navbtn" id="hbtn">Home</button>
        <button onClick={()=>move('/About')}className="navbtn">About</button>
        <button onClick={()=>move('/Contact')}className="navbtn">Contact</button>
        <button onClick={handleLogout}className="navbtn" id="logbtn">Logout</button>
      </div>

      {/* Hamburger icon visible only on mobile */}
      <div className="hamburger" onClick={toggleMenu}>
        ☰
      </div>

      {/* Mobile dropdown menu */}
      {open && (
        <div className="mobile-menu">
          <button onClick={()=>move('/Mainpage')} className="navbtn">Home</button>
          <button onClick={()=>move('/About')}className="navbtn">About</button>
          <button onClick={()=>move('/Contact')}className="navbtn">Contact</button>
          <button onClick={handleLogout}className="navbtn" id="logbtn">Logout</button>
        </div>
      )}
    </nav>
  );
}

export default Navbar;
