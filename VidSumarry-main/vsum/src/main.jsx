
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.jsx'
import Arrow from './assets/components/Arrow.jsx'
import { StrictMode } from 'react'
import Circle from './assets/components/circle/Circle.jsx'


createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <App />
    {/* <Arrow /> */}
  </BrowserRouter>

)
