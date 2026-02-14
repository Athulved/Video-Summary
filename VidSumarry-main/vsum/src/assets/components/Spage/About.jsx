import React from 'react'
import Navbar from './Navbar'
import './About.css'
const About = () => {
    return (
        <div className='about'>
            <Navbar />
            <p className='ptag'>ABOUT</p>
            
            <div className="content">
            In today’s fast-paced world, long videos can feel overwhelming. That’s where BreeZip comes in. We transform lengthy videos into concise, engaging summaries, delivering the key moments without the clutter. Whether it’s a Travel Documentary, a car review, or a documentary, BreeZip helps you watch more in less time.               
            </div>
            
        </div>
    )
}

export default About