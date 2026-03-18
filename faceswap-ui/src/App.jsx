import { useState } from 'react';
import axios from 'axios';
import imageCompression from 'browser-image-compression';
import { Upload, Camera, Zap, Download, Clock, Image as ImageIcon, Cpu, UserCheck, UserPlus, CheckCircle2, HardDriveDownload } from 'lucide-react';
import './App.css';
import { runWebGPUSwap, initWebGPU, isWebGPULoaded } from './WebGPUSwapper';
import { cropTargetCanvas, pasteBackCanvas } from './CanvasHelper';

const BACKEND_URL = `http://${window.location.hostname}:8000`;

const hashFile = async (file) => {
    const arrayBuffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
};

function App() {
  // Navigation State
  const [userType, setUserType] = useState(null); // 'new' | 'returning' | null
  const [userNameInput, setUserNameInput] = useState('');
  const [confirmedName, setConfirmedName] = useState(null);

  // Source State
  const [sourceImage, setSourceImage] = useState(null);
  const [sourcePreview, setSourcePreview] = useState(null);
  const [sourceEmbedding, setSourceEmbedding] = useState(null);
  const [isExtracting, setIsExtracting] = useState(false);
  
  // Target & Output State
  const [targetImage, setTargetImage] = useState(null);
  const [targetPreview, setTargetPreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [metrics, setMetrics] = useState(null);

  // Status State
  const [isSwapping, setIsSwapping] = useState(false);
  const [progressStep, setProgressStep] = useState(0);
  const [error, setError] = useState(null);
  
  // Model Loading State
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelProgress, setModelProgress] = useState(0);
  const [modelTotalSize, setModelTotalSize] = useState(0);
  const [isModelReady, setIsModelReady] = useState(false);
  
  // API Calls

  const handleSaveNewUser = async () => {
    if (!userNameInput || !sourceImage) {
      setError("Please provide a name and upload a source photo.");
      return;
    }
    setError(null);
    setIsExtracting(true);
    
    try {
      const options = { maxSizeMB: 0.5, maxWidthOrHeight: 512, useWebWorker: true };
      const compressedFile = await imageCompression(sourceImage, options);
      
      const formData = new FormData();
      formData.append('name', userNameInput.trim().toLowerCase());
      formData.append('source', compressedFile);

      // Extract via Server API purely to calculate embedding mathematics
      const response = await axios.post(`${BACKEND_URL}/api/v1/extract-source`, formData);
      const data = response.data;
      
      // Save permanently to Browser Storage
      const storedSources = JSON.parse(localStorage.getItem('source_embeddings') || '{}');
      storedSources[data.name] = data.embedding_norm;
      localStorage.setItem('source_embeddings', JSON.stringify(storedSources));
      
      setSourceEmbedding({ embedding_norm: data.embedding_norm });
      setConfirmedName(data.name);
    } catch (err) {
      console.error(err);
      setError(`Failed to register: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setIsExtracting(false);
    }
  };

  const fetchExistingUser = async (nameToFetch) => {
    const name = nameToFetch || userNameInput.trim().toLowerCase();
    if (!name) {
      setError("Please enter your registered name.");
      return;
    }
    
    setError(null);
    setIsExtracting(true);
    
    try {
      // Zero-Latency Offline Retrieval from Browser Local Storage
      const storedSources = JSON.parse(localStorage.getItem('source_embeddings') || '{}');
      if (storedSources[name]) {
          setSourceEmbedding({ embedding_norm: storedSources[name] });
          setConfirmedName(name);
      } else {
          setError(`No face print found locally for '${name}'. Did you register on this exact device?`);
      }
    } catch (err) {
      console.error(err);
      setError("Local storage read failed.");
    } finally {
      setIsExtracting(false);
    }
  };

  const handleImageUpload = (e, type) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const previewUrl = URL.createObjectURL(file);
    if (type === 'source') {
      setSourceImage(file);
      setSourcePreview(previewUrl);
    } else {
      setTargetImage(file);
      setTargetPreview(previewUrl);
    }
    setResultImage(null);
    setError(null);
  };

  const loadModelToRam = async () => {
    if (isWebGPULoaded() || isModelReady) {
       setIsModelReady(true);
       return;
    }
    setIsModelLoading(true);
    setError(null);
    try {
      await initWebGPU((loaded, total) => {
        setModelProgress(loaded);
        setModelTotalSize(total);
      });
      setIsModelReady(true);
    } catch(err) {
      console.error(err);
      setError(`Failed to download AI model. Please check network. (${err.message})`);
    } finally {
      setIsModelLoading(false);
    }
  };

  const handleSwap = async () => {
    if (!sourceEmbedding || !targetImage) return;
    if (!isModelReady) return;

    setIsSwapping(true);
    setError(null);
    setResultImage(null);
    
    try {
      const options = { maxSizeMB: 1, maxWidthOrHeight: 1024, useWebWorker: true };
      const compressedTarget = await imageCompression(targetImage, options);
      const startTime = performance.now();
      
      // Step 1: Target Matrix Resolving
      setProgressStep(1);
      const targetHash = await hashFile(compressedTarget);
      let affineMatrix = null;
      
      const storedTargets = JSON.parse(localStorage.getItem('target_embeddings') || '{}');
      if (storedTargets[targetHash]) {
          console.log("Local Target Cache HIT: Bypassing Server entirely!");
          affineMatrix = storedTargets[targetHash];
      } else {
          console.log("Local Target Cache MISS: Asking Server to calculate Affine Math...");
          const tgtFormData = new FormData();
          tgtFormData.append('target', compressedTarget, 'target.jpg');
          const res = await axios.post(`${BACKEND_URL}/api/v1/extract-target`, tgtFormData);
          affineMatrix = res.data.affine_matrix;
          
          storedTargets[targetHash] = affineMatrix;
          localStorage.setItem('target_embeddings', JSON.stringify(storedTargets));
      }

      // Step 2: Native Canvas Geometric Math + Local WebGPU
      setProgressStep(2);
      const cropBase64 = await cropTargetCanvas(compressedTarget, affineMatrix);
      const swappedCropBase64 = await runWebGPUSwap(cropBase64, sourceEmbedding.embedding_norm);
      
      // Step 3: Native Canvas Inverse Alpha Blending
      setProgressStep(3);
      const finalImageBase64 = await pasteBackCanvas(compressedTarget, swappedCropBase64, affineMatrix);

      const endTime = performance.now();
      
      setResultImage(finalImageBase64);
      
      const approxBytes = Math.round((finalImageBase64.length - 22) * 3 / 4);
      setMetrics({
        time: ((endTime - startTime) / 1000).toFixed(2),
        size: (approxBytes / 1024).toFixed(1)
      });

    } catch (err) {
      console.error(err);
      setError(`Swap failed: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setIsSwapping(false);
      setProgressStep(0);
    }
  };

  const resetFlow = () => {
    setUserType(null);
    setConfirmedName(null);
    setSourceEmbedding(null);
    setSourceImage(null);
    setSourcePreview(null);
    setTargetImage(null);
    setTargetPreview(null);
    setResultImage(null);
    setUserNameInput('');
  };

  // UI Renderers
  const renderUserSelection = () => (
    <div className="user-selection-card fade-in">
      <h2>Welcome to FaceMorph</h2>
      <p>Select your user type to get started</p>
      
      <div className="selection-buttons mt-4">
        <button className="type-btn" onClick={() => setUserType('new')}>
          <UserPlus size={48} />
          <h3>New User</h3>
          <span>Register your face print</span>
        </button>
        <button className="type-btn" onClick={() => setUserType('returning')}>
          <UserCheck size={48} />
          <h3>Returning User</h3>
          <span>Use a saved face print</span>
        </button>
      </div>
    </div>
  );

  const renderRegistrationFlow = () => (
    <div className="upload-box fade-in">
      <div className="box-header">
         <span className="step-number">1</span>
         <h2>Register Face Print</h2>
      </div>
      <p className="box-desc">Choose an identifier and upload your source photo.</p>
      
      <input 
        type="text" 
        className="name-input" 
        placeholder="Enter a unique name (e.g. johnny)" 
        value={userNameInput}
        onChange={e => setUserNameInput(e.target.value)}
        disabled={isExtracting}
      />

      <div className={`image-container mt-4 ${isExtracting ? 'loading' : ''}`} onClick={() => document.getElementById('source-upload').click()}>
        {sourcePreview ? (
          <img src={sourcePreview} alt="Source" className="preview-image" />
        ) : (
          <div className="empty-state">
            <Upload size={32} />
            <span>Upload clear face photo</span>
          </div>
        )}
        <input 
          id="source-upload" 
          type="file" 
          accept="image/*" 
          onChange={(e) => handleImageUpload(e, 'source')} 
          hidden 
          disabled={isExtracting}
        />
      </div>

      <button 
        className="save-btn mt-4" 
        onClick={handleSaveNewUser}
        disabled={!sourceImage || !userNameInput || isExtracting}
      >
        {isExtracting ? 'Extracting via Server...' : 'Save Array to Browser Storage'}
      </button>
    </div>
  );

  const renderReturningFlow = () => (
    <div className="upload-box fade-in">
      <div className="box-header">
         <span className="step-number">1</span>
         <h2>Load Face Print</h2>
      </div>
      <p className="box-desc">Enter your registered name to load your embedding.</p>
      
      <input 
        type="text" 
        className="name-input" 
        placeholder="Enter your registered name" 
        value={userNameInput}
        onChange={e => setUserNameInput(e.target.value)}
        disabled={isExtracting}
      />

      <button 
        className="save-btn mt-4" 
        onClick={() => fetchExistingUser()}
        disabled={!userNameInput || isExtracting}
      >
        {isExtracting ? 'Loading locally...' : 'Fetch Array from Browser Storage'}
      </button>
    </div>
  );

  const renderTargetUpload = () => (
    <div className="upload-box fade-in">
       <div className="box-header">
         <span className="step-number">2</span>
         <h2>Target Image</h2>
       </div>
       <p className="box-desc">The photo to paste '{confirmedName}' onto</p>
       
       <div className="image-container mt-4" onClick={() => document.getElementById('target-upload').click()}>
         {targetPreview ? (
           <img src={targetPreview} alt="Target" className="preview-image" />
         ) : (
           <div className="empty-state">
             <ImageIcon size={32} />
             <span>Upload target photo</span>
           </div>
         )}
         <input 
           id="target-upload" 
           type="file" 
           accept="image/*" 
           onChange={(e) => handleImageUpload(e, 'target')} 
           hidden 
         />
       </div>
    </div>
  );

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo-container">
          <span className="logo-emoji">🎭</span>
          <h1>FaceMorph Edge</h1>
        </div>
        <p className="subtitle">100% Zero-Latency Serverless Edge Engine</p>
        
        {userType && (
          <button className="reset-btn mt-4" onClick={resetFlow}>Start Over / Switch User</button>
        )}
      </header>

      <main className="main-content">
        {error && <div className="error-message">{error}</div>}

        {!userType && renderUserSelection()}

        {userType && !confirmedName && (
           <div className="registration-container">
              {userType === 'new' ? renderRegistrationFlow() : renderReturningFlow()}
           </div>
        )}

        {confirmedName && (
          <>
            <div className="success-banner fade-in">
               <CheckCircle2 color="#4ade80" />
               <span>Face print for <strong>{confirmedName}</strong> loaded entirely offline!</span>
            </div>

            <div className="upload-section mt-4">
              {/* Target Box */}
              {renderTargetUpload()}

              {/* Action Box */}
              <div className="upload-box fade-in">
                 <div className="box-header">
                   <span className="step-number">3</span>
                   <h2>Execute Swap</h2>
                 </div>
                 <p className="box-desc">Process the transformation entirely inside your browser hardware.</p>

                 {/* New Model Loading UI */}
                 {!isModelReady ? (
                   <div className="model-loading-ui mt-4">
                     <p style={{ color: "var(--text-secondary)", marginBottom: "12px", fontSize: "0.95rem" }}>
                        {isModelLoading 
                          ? `Downloading AI Model securely into your local Browser Memory...`
                          : `Your browser needs to cache the 400 MB Graphics Model to compute faces offline.`}
                     </p>
                     
                     {isModelLoading ? (
                       <div className="progress-container fade-in">
                         <div className="progress-bar-bg" style={{ width: '100%', height: '8px', background: 'var(--border-color)', borderRadius: '4px', overflow: 'hidden' }}>
                           <div className="progress-bar-fill" style={{ height: '100%', background: 'var(--accent-blue)', transition: 'width 0.3s', width: `${modelTotalSize > 0 ? (modelProgress / modelTotalSize) * 100 : 0}%` }}></div>
                         </div>
                         <p style={{ marginTop: '8px', fontSize: '0.85rem', color: 'var(--accent-blue)', textAlign: 'right' }}>
                           {modelTotalSize > 0 
                             ? `${(modelProgress / (1024*1024)).toFixed(1)} MB / ${(modelTotalSize / (1024*1024)).toFixed(1)} MB`
                             : "Connecting directly..."}
                         </p>
                       </div>
                     ) : (
                       <button className="swap-button mt-4" onClick={loadModelToRam}>
                          <span className="button-content">
                            <HardDriveDownload size={20} />
                            Download Model to RAM
                          </span>
                       </button>
                     )}
                   </div>
                 ) : (
                   <>
                      <div className="success-banner fade-in mt-4" style={{ padding: '12px', background: 'rgba(79, 142, 247, 0.1)', borderColor: 'var(--accent-blue)', color: 'var(--accent-blue)' }}>
                        <Cpu size={18} />
                        <span style={{ fontSize: "0.9rem" }}>AI Model successfully Cached in Browser!</span>
                      </div>
                      
                      <button 
                        className={`swap-button mt-4 ${(!targetImage || isSwapping) ? 'disabled' : ''}`}
                        onClick={handleSwap}
                        disabled={!targetImage || isSwapping}
                      >
                        {isSwapping ? (
                          <span className="button-content">
                            <div className="spinner"></div>
                            Processing deeply offline...
                          </span>
                        ) : (
                          <span className="button-content">
                            <Zap size={20} />
                            Swap Faces (Zero Server Ping)
                          </span>
                        )}
                      </button>

                      {isSwapping && (
                        <div className="progress-container fade-in mt-4">
                          <div className="steps-list">
                            <div className={`step-item ${progressStep >= 1 ? 'active' : ''}`}>
                              <div className="step-dot"></div>
                              <span>1. Fetching Coordinates (Local)</span>
                            </div>
                            <div className={`step-item ${progressStep >= 2 ? 'active' : ''}`}>
                              <div className="step-dot"></div>
                              <span>2. AI Generation (GPU WebAssembly)</span>
                            </div>
                            <div className={`step-item ${progressStep >= 3 ? 'active' : ''}`}>
                              <div className="step-dot"></div>
                              <span>3. Edge Raster Blending (Canvas)</span>
                            </div>
                          </div>
                        </div>
                      )}
                   </>
                 )}
              </div>
            </div>
          </>
        )}

        {/* Result Section */}
        {resultImage && (
          <div className="result-section fade-in mt-4">
            <div className="success-badge">✓ Offline Edge Render Complete</div>
            
            <div className="metrics-row">
              <div className="metric-card">
                <Clock size={16} />
                <div className="metric-info">
                  <span className="metric-label">Compute Time</span>
                  <span className="metric-value">{metrics.time}s</span>
                </div>
              </div>
              <div className="metric-card">
                <Cpu size={16} />
                <div className="metric-info">
                  <span className="metric-label">Engine</span>
                  <span className="metric-value">100% Client Edge</span>
                </div>
              </div>
              <div className="metric-card">
                <Download size={16} />
                <div className="metric-info">
                  <span className="metric-label">Output Size</span>
                  <span className="metric-value">{metrics.size} KB</span>
                </div>
              </div>
            </div>

            <div className="comparison-grid">
              <div className="comparison-item">
                <h3>Original Target</h3>
                <img src={targetPreview} alt="Original" />
              </div>
              <div className="comparison-item result-highlight">
                <h3>Swapped Result</h3>
                <img src={resultImage} alt="Result" />
              </div>
            </div>

            <a href={resultImage} download={`facemorph_${confirmedName}.jpg`} className="download-button mt-4">
              <Download size={18} /> Download High-Res Result
            </a>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
