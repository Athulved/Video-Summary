import React from 'react'
import './App.css'
import { Route, Router, Routes } from 'react-router-dom'
import Circle from './assets/components/circle/Circle'
import Spage from './assets/components/Spage/Spage'
import About from './assets/components/Spage/About'
import Contact from './assets/components/Spage/Contact'
import Login from "./assets/components/login";
import Register from "./assets/components/register";

const App = () => {
  return (
    // <div className='app'>
    //   <div class='ripple-background'>
    //     <div class='circle xxlarge shade1'></div>
    //     <div class='circle xlarge shade2'></div>
    //     <div class='circle large shade3'></div>
    //     <div class='circle mediun shade4'></div>
    //     <div class='circle small shade5'></div>
    //   </div>
    <div>
      <Routes>
        <Route path="/" element={<Circle/>}></Route>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path='/Mainpage' element={<Spage/>}></Route>
        <Route path='/About' element={<About/>}></Route>
        <Route path='/Contact' element={<Contact/>}></Route>
      </Routes>
    </div>





  )
}

export default App
