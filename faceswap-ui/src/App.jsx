import { useState, useEffect } from 'react';
import axios from 'axios';
import imageCompression from 'browser-image-compression';
import { Upload, Camera, Zap, Download, Clock, Image as ImageIcon, Cpu, UserCheck, UserPlus, CheckCircle2, HardDriveDownload, X } from 'lucide-react';
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
  const [showSuccessBanner, setShowSuccessBanner] = useState(false);

  // Source State
  const [sourceImage, setSourceImage] = useState(null);
  const [sourcePreview, setSourcePreview] = useState(null);
  const [sourceEmbedding, setSourceEmbedding] = useState(null);
  const [userSkinTone, setUserSkinTone] = useState(null);
  const [isExtracting, setIsExtracting] = useState(false);
  
  // Target & Output State
  const [targetImage, setTargetImage] = useState(null);
  const [targetPreview, setTargetPreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [metrics, setMetrics] = useState(null);

  // Status State
  const [isSwapping, setIsSwapping] = useState(false);
  const [progressStep, setProgressStep] = useState(0);
  const [error, setError] = useState(null);
  
  // Auto-hide error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        setError(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);
  
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
      
      const colorRes = await axios.post(`${BACKEND_URL}/api/v1/extract-user-tone`, formData);
      const skinTone = colorRes.data.skin_tone;
      
      // Save permanently to Browser Storage
      const storedSources = JSON.parse(localStorage.getItem('source_embeddings') || '{}');
      storedSources[data.name] = data.embedding_norm;
      localStorage.setItem('source_embeddings', JSON.stringify(storedSources));
      
      const storedTones = JSON.parse(localStorage.getItem('user_tones_v2') || '{}');
      storedTones[data.name] = skinTone;
      localStorage.setItem('user_tones_v2', JSON.stringify(storedTones));
      
      setSourceEmbedding({ embedding_norm: data.embedding_norm });
      setUserSkinTone(skinTone);
      setConfirmedName(data.name);
      setShowSuccessBanner(true);
      setTimeout(() => setShowSuccessBanner(false), 7000);
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
      const storedTones = JSON.parse(localStorage.getItem('user_tones_v2') || '{}');
      if (storedSources[name]) {
          setSourceEmbedding({ embedding_norm: storedSources[name] });
          setUserSkinTone(storedTones[name] || null);
          setConfirmedName(name);
          setShowSuccessBanner(true);
          setTimeout(() => setShowSuccessBanner(false), 7000);
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
      // Step 1: Securely Hash the UNCOMPRESSED Target File (User's Core Idea!)
      // This is incredibly smart. Hashing the raw file guarantees absolute cache uniqueness
      // without relying on browser-specific Canvas Compression variations.
      setProgressStep(1);
      const targetHash = await hashFile(targetImage);
      
      const options = { maxSizeMB: 1, maxWidthOrHeight: 1024, useWebWorker: true };
      const compressedTarget = await imageCompression(targetImage, options);
      
      const startTime = performance.now();
      let affineMatrix = null;
      
      const storedTargets = JSON.parse(localStorage.getItem('target_embeddings') || '{}');
      if (storedTargets[targetHash]) {
          console.log("Local Target Cache HIT: Exact File Matched! Bypassing Server...");
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

  const calculateDistance = (vec1, vec2) => {
    const wMean = 1.0;
    const wStd = 0.5;
    return Math.sqrt(
      wMean * (Math.pow(vec1[0] - vec2[0], 2) + Math.pow(vec1[1] - vec2[1], 2) + Math.pow(vec1[2] - vec2[2], 2)) +
      wStd * (Math.pow(vec1[3] - vec2[3], 2) + Math.pow(vec1[4] - vec2[4], 2) + Math.pow(vec1[5] - vec2[5], 2))
    );
  };

  const STOCK_PATHS = [
    '/stocks/Fair/model1.jpg', '/stocks/Fair/model2.jpg', '/stocks/Fair/model3.jpg', '/stocks/Fair/model4.jpg', '/stocks/Fair/model5.jpg',
    '/stocks/Dusky/model1.jpg', '/stocks/Dusky/model2.jpg', '/stocks/Dusky/model3.jpg', '/stocks/Dusky/model4.jpg', '/stocks/Dusky/model5.jpg',
    '/stocks/Dark/model1.jpg', '/stocks/Dark/model2.jpg', '/stocks/Dark/model3.jpg', '/stocks/Dark/model4.jpg', '/stocks/Dark/model5.jpg',
  ];

  const handleAutoMatchSwap = async () => {
    if (!userSkinTone || !sourceEmbedding || !isModelReady) return;

    setIsSwapping(true);
    setError(null);
    setBatchResults([]);
    setResultImage(null);

    try {
      setProgressStep(1);
      let stockTones = JSON.parse(localStorage.getItem('stock_tones_v2'));
      if (!stockTones || Object.keys(stockTones).length < 15) {
         console.log("Local Stock Cache MISS: Asking Server for batch encodings...");
         const res = await axios.post(`${BACKEND_URL}/api/v1/extract-stock-tones`, { paths: STOCK_PATHS });
         stockTones = res.data.stock_tones;
         localStorage.setItem('stock_tones_v2', JSON.stringify(stockTones));
      }

      let bestFolder = null;
      let minAvgDistance = Infinity;
      const folders = ['Fair', 'Dusky', 'Dark'];
      
      for (let folder of folders) {
         const folderPaths = STOCK_PATHS.filter(p => p.includes(`/${folder}/`));
         let totalDist = 0;
         let count = 0;
         for (let path of folderPaths) {
             const tone = stockTones[path];
             if (tone) {
                 totalDist += calculateDistance(userSkinTone, tone);
                 count++;
             }
         }
         if (count > 0) {
             const avg = totalDist / count;
             if (avg < minAvgDistance) {
                 minAvgDistance = avg;
                 bestFolder = folder;
             }
         }
      }

      setProgressStep(2);
      const targetPaths = STOCK_PATHS.filter(p => p.includes(`/${bestFolder}/`));
      const startTime = performance.now();
      let newBatchResults = [];

      for (let i = 0; i < targetPaths.length; i++) {
          const path = targetPaths[i];
          setError(`Swapping image ${i+1} of ${targetPaths.length} from ${bestFolder}...`);
          
          const response = await fetch(path);
          const blob = await response.blob();
          const targetFile = new File([blob], path.split('/').pop(), { type: blob.type });

          const targetHash = await hashFile(targetFile);
          const options = { maxSizeMB: 1, maxWidthOrHeight: 1024, useWebWorker: true };
          const compressedTarget = await imageCompression(targetFile, options);

          let affineMatrix = null;
          const storedTargets = JSON.parse(localStorage.getItem('target_embeddings') || '{}');
          if (storedTargets[targetHash]) {
              affineMatrix = storedTargets[targetHash];
          } else {
              const tgtFormData = new FormData();
              tgtFormData.append('target', compressedTarget, 'target.jpg');
              const res = await axios.post(`${BACKEND_URL}/api/v1/extract-target`, tgtFormData);
              affineMatrix = res.data.affine_matrix;
              storedTargets[targetHash] = affineMatrix;
              localStorage.setItem('target_embeddings', JSON.stringify(storedTargets));
          }

          const cropBase64 = await cropTargetCanvas(compressedTarget, affineMatrix);
          const swappedCropBase64 = await runWebGPUSwap(cropBase64, sourceEmbedding.embedding_norm);
          const finalImageBase64 = await pasteBackCanvas(compressedTarget, swappedCropBase64, affineMatrix);
          
          newBatchResults.push(finalImageBase64);
          setBatchResults([...newBatchResults]);
      }

      const endTime = performance.now();
      setError(null);
      setMetrics({
        time: ((endTime - startTime) / 1000).toFixed(2),
        size: "Batch"
      });
    } catch (err) {
      console.error(err);
      setError(`Auto-Match Swap failed: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setIsSwapping(false);
      setProgressStep(0);
    }
  };

  const resetFlow = () => {
    setUserType(null);
    setConfirmedName(null);
    setSourceEmbedding(null);
    setUserSkinTone(null);
    setSourceImage(null);
    setSourcePreview(null);
    setTargetImage(null);
    setTargetPreview(null);
    setResultImage(null);
    setBatchResults([]);
    setUserNameInput('');
  };

  // UI Renderers
  const renderUserSelection = () => (
    <div className="user-selection-card fade-in">
      <h2>Welcome to FaceSwap</h2>
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
    <div className="upload-box fade-in" style={{ maxWidth: '500px', margin: '0 auto' }}>
      <div className="box-header" style={{ justifyContent: 'center' }}>
         <h2>Register Face Print</h2>
      </div>
      <p className="box-desc" style={{ textAlign: 'center' }}>Choose an identifier and upload your source photo.</p>
      
      <input 
        type="text" 
        className="name-input" 
        placeholder="Enter your name" 
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
        style={{ background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))', border: 'none' }}
        onClick={handleSaveNewUser}
        disabled={!sourceImage || !userNameInput || isExtracting}
      >
        {isExtracting ? 'Uploading...' : 'Upload'}
      </button>
    </div>
  );

  const renderReturningFlow = () => (
    <div className="upload-box fade-in" style={{ maxWidth: '500px', margin: '0 auto' }}>
      <div className="box-header" style={{ justifyContent: 'center' }}>
         <h2>Load Face Print</h2>
      </div>
      <p className="box-desc" style={{ textAlign: 'center' }}>Enter your registered name</p>
      
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
        style={{ background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))', border: 'none' }}
        onClick={() => fetchExistingUser()}
        disabled={!userNameInput || isExtracting}
      >
        {isExtracting ? 'Loading locally...' : 'Fetch Face Print'}
      </button>
    </div>
  );

  // Target Upload features intentionally removed for strictly automated target matching

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo-container">
         
          <h1>FaceSwap</h1>
        </div>
        
       {userType && (
          <div className="action-buttons mt-4" style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
            <button className="reset-btn" onClick={resetFlow}>Start Over</button>
            <button className="reset-btn" onClick={() => {
              resetFlow();
              setUserType('returning');
            }}>Switch User</button>
          </div>
        )}
      </header>

      <main className="main-content">
        {error && (
          <div className="error-message fade-in" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>{error}</span>
            <button onClick={() => setError(null)} style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '4px' }}>
              <X size={18} />
            </button>
          </div>
        )}

        {!userType && renderUserSelection()}

        {userType && !confirmedName && (
           <div className="registration-container">
              {userType === 'new' ? renderRegistrationFlow() : renderReturningFlow()}
           </div>
        )}

        {confirmedName && (
          <>
            {showSuccessBanner && (
              <div className="success-banner fade-in" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                 <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                   <CheckCircle2 color="#4ade80" />
                   <span>Face print for <strong>{confirmedName}</strong> loaded entirely offline!</span>
                 </div>
                 <button onClick={() => setShowSuccessBanner(false)} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                   <X size={18} />
                 </button>
              </div>
            )}

            <div className="upload-section mt-4">
              {/* Action Box */}
              <div className="upload-box fade-in" style={{ maxWidth: '600px', margin: '0 auto' }}>
                 <div className="box-header" style={{ justifyContent: 'center' }}>
                   <h2>Execute Auto-Match</h2>
                 </div>
                 <p className="box-desc" style={{ textAlign: 'center', marginBottom: '1rem' }}>Automatically match skin tone and swap all 5 folder styles</p>

                 {/* New Model Loading UI */}
                 {!isModelReady ? (
                   <div className="model-loading-ui mt-4">
                     {isModelLoading && (
                       <p style={{ color: "var(--text-secondary)", marginBottom: "12px", fontSize: "0.95rem" }}>
                          Downloading...
                       </p>
                     )}
                     
                     {isModelLoading ? (
                       <div className="progress-container fade-in">
                         <div className="progress-bar-bg" style={{ width: '100%', height: '8px', background: 'var(--border-color)', borderRadius: '4px', overflow: 'hidden' }}>
                           <div className="progress-bar-fill" style={{ height: '100%', background: 'var(--accent-blue)', transition: 'width 0.3s', width: `${modelTotalSize > 0 ? (modelProgress / modelTotalSize) * 100 : 0}%` }}></div>
                         </div>
                         <p style={{ marginTop: '8px', fontSize: '0.85rem', color: 'var(--accent-blue)', textAlign: 'right' }}>
                           {modelTotalSize > 0 
                             ? `${((modelProgress / modelTotalSize) * 100).toFixed(0)} %`
                             : "Connecting directly..."}
                         </p>
                       </div>
                     ) : (
                       <button className="swap-button mt-4" onClick={loadModelToRam}>
                          <span className="button-content">
                            <HardDriveDownload size={20} />
                           Get started with FaceSwap
                          </span>
                       </button>
                     )}
                   </div>
                 ) : (
                   <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <button 
                        className={`swap-button ${(isSwapping || !userSkinTone) ? 'disabled' : ''}`}
                        onClick={handleAutoMatchSwap}
                        style={{ background: 'linear-gradient(135deg, #10b981, #059669)' }}
                        disabled={isSwapping || !userSkinTone}
                      >
                        {isSwapping ? (
                          <span className="button-content">
                            <div className="spinner"></div>
                            Auto-Matching...
                          </span>
                        ) : (
                          <span className="button-content">
                            <Zap size={20} />
                            Auto Match & Batch Swap
                          </span>
                        )}
                      </button>
                   </div>
                 )}
              </div>
            </div>
          </>
        )}

        {/* Result Section */}
        {(resultImage || batchResults.length > 0) && (
          <div className="result-section fade-in mt-4">
            <div className="success-badge">✓ Offline Edge Render Complete</div>
            
            {metrics && (
              <div className="metrics-row">
                <div className="metric-card">
                  <Clock size={16} />
                  <div className="metric-info">
                    <span className="metric-label">Compute Time</span>
                    <span className="metric-value">{metrics.time}s</span>
                  </div>
                </div>
                <div className="metric-card">
                  <Download size={16} />
                  <div className="metric-info">
                    <span className="metric-label">Output Size</span>
                    <span className="metric-value">
                      {metrics.size} {metrics.size !== 'Batch' ? 'KB' : ''}
                    </span>
                  </div>
                </div>
              </div>
            )}
<div className="comparison-grid">
              {sourcePreview && (
                <div className="comparison-item">
                  <h3>Source Image</h3>
                  <img src={sourcePreview} alt="Source" />
                </div>
              )}
              {resultImage && (
                <div className="comparison-item result-highlight">
                  <h3>Swapped Result</h3>
                  <img src={resultImage} alt="Result" />
                </div>
              )}
            </div>

            {batchResults.length > 0 && (
              <div className="batch-results mt-4">
                <h3 style={{ marginBottom: '1rem' }}>Batch Swap Results</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                  {batchResults.map((res, i) => (
                    <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <img src={res} alt={`Batch Result ${i+1}`} style={{ width: '100%', borderRadius: '8px' }} />
                      <a href={res} download={`facemorph_${confirmedName}_style_${i+1}.jpg`} className="download-button" style={{ display: 'flex', padding: '8px', justifyContent: 'center', fontSize: '14px', textDecoration: 'none' }}>
                        <Download size={16} style={{ marginRight: '6px' }} /> Save Image
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {resultImage && (
              <a href={resultImage} download={`facemorph_${confirmedName}.jpg`} className="download-button mt-4">
                <Download size={18} /> Download High-Res Result
              </a>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;