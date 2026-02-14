
import React, { useState, useEffect, useRef } from 'react';
import './VideoUpload.css';
import cloud from './cloud.svg';

function VideoUpload() {
  const [file, setFile] = useState(null);
  const [thumbnail, setThumbnail] = useState(null);
  const [summary, setSummary] = useState('');
  const [summaryText, setSummaryText] = useState('');
  const [summaryAudio, setSummaryAudio] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [youtubeLink, setYoutubeLink] = useState('');
  const [youtubePreview, setYoutubePreview] = useState('');
  const [sliderValue, setSliderValue] = useState(20);

  const svideoRef = useRef(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const videoURL = URL.createObjectURL(selectedFile);
      setThumbnail(videoURL);
      setYoutubePreview('');
      setYoutubeLink('');
      setSummary('');
      setSummaryText('');
      setSummaryAudio('');
    }
  };

  const handleYoutubeLinkChange = (event) => {
    const link = event.target.value;
    setYoutubeLink(link);

    const videoId = extractYouTubeId(link);
    if (videoId) {
      setYoutubePreview(`https://www.youtube.com/embed/${videoId}`);
      setFile(null);
      setThumbnail('');
      setSummary('');
      setSummaryText('');
      setSummaryAudio('');
    } else {
      setYoutubePreview('');
    }
  };

  const extractYouTubeId = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return match && match[2].length === 11 ? match[2] : null;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!file && !youtubeLink) {
      alert('Please select a video file or enter a YouTube link.');
      return;
    }

    setIsProcessing(true);
    setSummary('');
    setSummaryText('');
    setSummaryAudio('');

    const formData = new FormData();
    if (file) {
      formData.append('video', file);
      formData.append('compression', sliderValue); // Add slider value to form data
    } else if (youtubeLink) {
      formData.append('youtube_link', youtubeLink);
      formData.append('compression', sliderValue); // Add slider value to form data
    }

    try {
      const response = await fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setSummary(`http://127.0.0.1:5000/summaries/${data.summary_video}`);
        setSummaryText(data.summary_text);
        setSummaryAudio(`http://127.0.0.1:5000/audio_summaries/${data.summary_audio}`);
      } else {
        console.error('Backend Error:', response.statusText);
        setSummaryText('Error: Unable to summarize text.');
      }
    } catch (error) {
      console.error('Fetch Error:', error);
      setSummaryText('Error: ' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSliderChange = (event) => {
    setSliderValue(event.target.value);
  };

  useEffect(() => {
    if (summary && svideoRef.current) {
      svideoRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [summary]);

  return (
    <div className="center-container">
      {file ? (
        <div className={`video-preview ${summary ? 'video-preview-adjusted' : ''}`}>
          <video key={thumbnail} width="200" controls>
            <source src={thumbnail} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          <p>{file.name}</p>
        </div>
      ) : youtubePreview ? (
        <div className={`video-preview ${summary ? 'video-preview-adjusted' : ''}`}>
          <iframe
            width="320"
            height="180"
            src={youtubePreview}
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          ></iframe>
        </div>
      ) : (
        <div className='image-div'>
          <img src={cloud} alt="Upload Icon" width="200" className="upload-icon" />
        </div>
      )}

      <br /><br />
      <form id="uploadForm" onSubmit={handleSubmit}>
        <div className='load-css'>
          <div className="button-container">
            <button type="button" id="chooseVideoButton" onClick={() => document.getElementById('videoInput').click()}>
              Choose Video File
            </button>
            <input type="file" id="videoInput" accept="video/*" style={{ display: 'none' }} onChange={handleFileChange} />
          </div>
          <div className='opt-text'>OR</div>
          <div className="youtube-link-container" style={{ marginTop: '20px' }}>
            <input
              type="text"
              placeholder="Paste YouTube link here"
              value={youtubeLink}
              onChange={handleYoutubeLinkChange}
              className="youtube-link-input"
            />
          </div>
        </div>

        <div className='load-css-b'>
          <div className="slider-container">
            <label htmlFor="compression-slider">Summary Percentage: {sliderValue}%</label>
            <input
              type="range"
              id="compression-slider"
              min="20"
              max="80"
              value={sliderValue}
              onChange={handleSliderChange}
              className="styled-slider"
            />
          </div>
          <div className="button-container" style={{ marginTop: '20px' }}>
            <button type="submit" id="summarizeButton" disabled={isProcessing}>
              {isProcessing ? "Processing..." : "Summarize Video"}
            </button>
          </div>
        </div>
      </form>

      {isProcessing && (
        <div className="progress-bar">
          <p>Processing Video...</p>
          <progress />
        </div>
      )}

      {/* {summary && !isProcessing && (
        <div className='load-css-a'>
          <div className="svideo" ref={svideoRef}>
            <h3>Summarized Video:</h3>
            <video controls width="500px">
              <source src={summary} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        </div>
      )} */}
      {summary && !isProcessing && (
        <div className='load-css-a'>
          <div className="svideo" ref={svideoRef}>
            <h3>Summarized Video :</h3>
            <video
              controls
              width="500px"
              key={`${summary}?v=${sliderValue}-${Date.now()}`}
            >
              <source
                src={`${summary}?v=${sliderValue}-${Date.now()}`}
                type="video/mp4"
              />
            </video>
          </div>
        </div>
      )}



      {summaryText && !isProcessing && (
        <div className='load-css-a'>
          <div className="text-summary">
            <h3>Summarized Text:</h3>
            <p>{summaryText}</p>
          </div>
        </div>
      )}

      {/* {summaryAudio && !isProcessing && (
        <div className='load-css-a'>
          <div className="audio-summary">
            <h3>Summarized Audio:</h3>
            <audio controls>
              <source src={summaryAudio} type="audio/mpeg" />
              Your browser does not support the audio element.
            </audio>
          </div>
        </div>
      )} */}
      {summaryAudio && !isProcessing && (
        <div className='load-css-a'>
          <div className="audio-summary">
            <h3>Summarized Audio:</h3>
            <audio controls key={`${summaryAudio}?v=${sliderValue}-${Date.now()}`}>
              <source
                src={`${summaryAudio}?v=${sliderValue}-${Date.now()}`}
                type="audio/mpeg"
              />
            </audio>
          </div>
        </div>
      )}
    </div>
  );
}

export default VideoUpload;

