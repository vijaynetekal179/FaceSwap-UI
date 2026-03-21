import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import imageCompression from 'browser-image-compression';
import { Upload, Camera, Zap, Download, Clock, Image as ImageIcon, Cpu, UserCheck, UserPlus, CheckCircle2, HardDriveDownload, X, Layers, Palette } from 'lucide-react';
import './App.css';
import { runWebGPUSwap, initWebGPU, isWebGPULoaded } from './WebGPUSwapper';
import { cropTargetCanvas, pasteBackCanvas } from './CanvasHelper';

const BACKEND_URL = "";

const hashFile = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
};

// Euclidean distance between two LAB arrays
const labDistance = (a, b) => Math.sqrt(
  a.reduce((sum, v, i) => sum + (v - b[i]) ** 2, 0)
);

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
  const [userSkinLab, setUserSkinLab] = useState(null);
  const [isExtracting, setIsExtracting] = useState(false);

  // Template State
  const [templates, setTemplates] = useState([]); // [{file, preview, skin_lab, affine_matrix, hash}]
  const [matchedTemplate, setMatchedTemplate] = useState(null);
  const [isLoadingTemplates, setIsLoadingTemplates] = useState(false);

  // Output State
  const [resultImage, setResultImage] = useState(null);
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
  const loadInitiated = useRef(false);

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

      // Extract via Server API — now returns embedding_norm + skin_lab
      const response = await axios.post(`${BACKEND_URL}/api/v1/extract-source`, formData);
      const data = response.data;

      // Save permanently to Browser Storage (now includes skin_lab)
      const storedSources = JSON.parse(localStorage.getItem('source_embeddings') || '{}');
      storedSources[data.name] = {
        embedding_norm: data.embedding_norm,
        skin_lab: data.skin_lab
      };
      localStorage.setItem('source_embeddings', JSON.stringify(storedSources));

      setSourceEmbedding({ embedding_norm: data.embedding_norm });
      setUserSkinLab(data.skin_lab);
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
      if (storedSources[name]) {
        const stored = storedSources[name];
        // Handle both old format (just array) and new format (object with skin_lab)
        if (Array.isArray(stored)) {
          setSourceEmbedding({ embedding_norm: stored });
          setUserSkinLab(null); // Old format, no skin_lab
        } else {
          setSourceEmbedding({ embedding_norm: stored.embedding_norm });
          setUserSkinLab(stored.skin_lab || null);
        }
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
    }
    setResultImage(null);
    setMatchedTemplate(null);
    setError(null);
  };

  // Handle template uploads (multiple files)
  const handleTemplateUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    setIsLoadingTemplates(true);
    setError(null);
    setResultImage(null);
    setMatchedTemplate(null);

    const templateCache = JSON.parse(localStorage.getItem('template_cache') || '{}');
    const newTemplates = [...templates];

    for (const file of files) {
      try {
        const hash = await hashFile(file);
        const preview = URL.createObjectURL(file);

        // Check localStorage cache
        if (templateCache[hash]) {
          console.log(`Template CACHE HIT: ${file.name} — skipping server call`);
          newTemplates.push({
            file: file,
            fileName: file.name,
            preview,
            hash,
            skin_lab: templateCache[hash].skin_lab,
            affine_matrix: templateCache[hash].affine_matrix
          });
          continue;
        }

        // Cache miss — send to server
        console.log(`Template CACHE MISS: ${file.name} — extracting from server...`);
        const options = { maxSizeMB: 1, maxWidthOrHeight: 1024, useWebWorker: true };
        const compressed = await imageCompression(file, options);

        const formData = new FormData();
        formData.append('template', compressed, file.name);

        const res = await axios.post(`${BACKEND_URL}/api/v1/extract-template`, formData);
        const data = res.data;

        // Save to localStorage
        templateCache[hash] = {
          skin_lab: data.skin_lab,
          affine_matrix: data.affine_matrix
        };
        localStorage.setItem('template_cache', JSON.stringify(templateCache));

        newTemplates.push({
          file: file,
          fileName: file.name,
          preview,
          hash,
          skin_lab: data.skin_lab,
          affine_matrix: data.affine_matrix
        });
      } catch (err) {
        console.error(`Failed to process template ${file.name}:`, err);
        setError(`Failed to extract template: ${file.name}. ${err?.response?.data?.detail || err.message}`);
      }
    }

    setTemplates(newTemplates);
    setIsLoadingTemplates(false);

    // Reset file input so the same files can be re-selected
    e.target.value = '';
  };

  const removeTemplate = (index) => {
    setTemplates(prev => prev.filter((_, i) => i !== index));
    setResultImage(null);
    setMatchedTemplate(null);
  };

  const loadModelToRam = async () => {
    if (isWebGPULoaded() || isModelReady || loadInitiated.current) {
      setIsModelReady(true);
      return;
    }
    loadInitiated.current = true;
    setIsModelLoading(true);
    setError(null);
    try {
      await initWebGPU((loaded, total) => {
        setModelProgress(loaded);
        setModelTotalSize(total);
      });
      setIsModelReady(true);
    } catch (err) {
      console.error(err);
      setError(`Failed to download AI model. Please check network. (${err.message})`);
    } finally {
      setIsModelLoading(false);
    }
  };

  // Background model loading on mount
  useEffect(() => {
    loadModelToRam();
  }, []);

  const handleSwap = async () => {
    if (!sourceEmbedding || templates.length === 0) return;
    if (!isModelReady) return;

    setIsSwapping(true);
    setError(null);
    setResultImage(null);

    try {
      // Step 1: Skin Tone Matching — pick the closest template
      setProgressStep(1);

      let bestTemplate;
      if (userSkinLab && templates.length > 1) {
        // Find template with smallest LAB distance
        let bestDist = Infinity;
        for (const t of templates) {
          const dist = labDistance(userSkinLab, t.skin_lab);
          console.log(`  Skin match: ${t.fileName} → distance=${dist.toFixed(2)}`);
          if (dist < bestDist) {
            bestDist = dist;
            bestTemplate = t;
          }
        }
        console.log(`✓ Best skin match: ${bestTemplate.fileName} (dist=${bestDist.toFixed(2)})`);
      } else {
        // Single template or no skin data — just use the first one
        bestTemplate = templates[0];
        console.log(`Using template: ${bestTemplate.fileName} (single/no skin data)`);
      }

      setMatchedTemplate(bestTemplate);

      // Step 2: Use cached affine matrix — crop + WebGPU swap
      setProgressStep(2);
      const startTime = performance.now();

      const affineMatrix = bestTemplate.affine_matrix;

      // Compress the template image for canvas processing
      const options = { maxSizeMB: 1, maxWidthOrHeight: 1024, useWebWorker: true };
      const compressedTarget = await imageCompression(bestTemplate.file, options);

      const cropBase64 = await cropTargetCanvas(compressedTarget, affineMatrix);
      const swappedCropBase64 = await runWebGPUSwap(cropBase64, sourceEmbedding.embedding_norm);

      // Step 3: Paste back
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
    setUserSkinLab(null);
    setSourceImage(null);
    setSourcePreview(null);
    setTemplates([]);
    setMatchedTemplate(null);
    setResultImage(null);
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

  const renderTemplateSection = () => (
    <div className="upload-box fade-in">
      <div className="box-header">
        <Layers size={20} />
        <h2>Template Models</h2>
      </div>
      <p className="box-desc">Upload template model images (different skin tones). Best match will be auto-selected.</p>

      {/* Upload Button */}
      <button
        className="template-upload-btn"
        onClick={() => document.getElementById('template-upload').click()}
        disabled={isLoadingTemplates}
      >
        {isLoadingTemplates ? (
          <span className="button-content"><div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></div> Processing...</span>
        ) : (
          <span className="button-content"><Upload size={18} /> Upload Templates</span>
        )}
      </button>
      <input
        id="template-upload"
        type="file"
        accept="image/*"
        multiple
        onChange={handleTemplateUpload}
        hidden
      />

      {/* Template Grid */}
      {templates.length > 0 && (
        <div className="template-grid mt-4">
          {templates.map((t, idx) => (
            <div
              key={t.hash}
              className={`template-card ${matchedTemplate?.hash === t.hash ? 'matched' : ''}`}
            >
              <button className="template-remove" onClick={() => removeTemplate(idx)}>
                <X size={14} />
              </button>
              <img src={t.preview} alt={t.fileName} />
              <div className="template-info">
                <span className="template-name">{t.fileName}</span>
                <span className="template-tone">
                  <Palette size={12} /> L={t.skin_lab[0].toFixed(0)}
                </span>
              </div>
              {matchedTemplate?.hash === t.hash && (
                <div className="match-badge">
                  <CheckCircle2 size={12} /> Best Match
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {userSkinLab && (
        <div className="user-tone-indicator mt-4">
          <Palette size={14} />
          <span>Your skin tone: L={userSkinLab[0].toFixed(1)}</span>
        </div>
      )}
    </div>
  );

  return (
    <div className="app-container">
      <header className="header" style={{ position: 'relative' }}>
        <div className="logo-container">
          <h1>FaceSwap</h1>
        </div>

        {isModelLoading && (
          <div style={{ position: 'absolute', right: '20px', top: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Initializing AI model: {modelTotalSize > 0 ? `${((modelProgress / modelTotalSize) * 100).toFixed(0)}%` : "Connecting..."}
            </span>
          </div>
        )}

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
                  <span>Face print for <strong>{confirmedName}</strong> loaded{userSkinLab ? ` • Skin tone: L=${userSkinLab[0].toFixed(0)}` : ''}!</span>
                </div>
                <button onClick={() => setShowSuccessBanner(false)} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  <X size={18} />
                </button>
              </div>
            )}

            <div className="upload-section mt-4">
              {/* Template Section */}
              {renderTemplateSection()}

              {/* Action Box */}
              <div className="upload-box fade-in">
                <div className="box-header">
                  <h2>Execute Swap</h2>
                </div>

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
                  <>
                    {/* Swap Status Info */}
                    {templates.length > 0 && (
                      <div className="swap-info mt-4">
                        <p><strong>{templates.length}</strong> template{templates.length > 1 ? 's' : ''} loaded</p>
                        {userSkinLab && templates.length > 1 && (
                          <p style={{ color: 'var(--accent-blue)', fontSize: '0.85rem' }}>
                            Skin tone matching will auto-select the best template
                          </p>
                        )}
                      </div>
                    )}

                    <button
                      className={`swap-button mt-4 ${(!templates.length || isSwapping) ? 'disabled' : ''}`}
                      onClick={handleSwap}
                      disabled={!templates.length || isSwapping}
                    >
                      {isSwapping ? (
                        <span className="button-content">
                          <div className="spinner"></div>
                          {progressStep === 1 ? 'Matching skin tone...' :
                            progressStep === 2 ? 'Swapping face...' :
                              progressStep === 3 ? 'Blending result...' : 'Processing...'}
                        </span>
                      ) : (
                        <span className="button-content">
                          <Zap size={20} />
                          Swap Faces
                        </span>
                      )}
                    </button>
                  </>
                )}
              </div>
            </div>
          </>
        )}

        {/* Result Section */}
        {resultImage && (
          <div className="result-section fade-in mt-4">
            <div className="success-badge">✓ Swap Complete</div>

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
                  <span className="metric-value">{metrics.size} KB</span>
                </div>
              </div>
            </div>

            {/* 3-Column Comparison: Source | Template | Result */}
            <div className="comparison-grid three-col">
              <div className="comparison-item">
                <h3>Source</h3>
                {sourcePreview ? (
                  <img src={sourcePreview} alt="Source" />
                ) : (
                  <div className="comparison-placeholder">
                    <span>{confirmedName}</span>
                  </div>
                )}
              </div>
              {matchedTemplate && (
                <div className="comparison-item template-highlight">
                  <h3>Matched Template</h3>
                  <img src={matchedTemplate.preview} alt="Template" />
                  <span className="match-label">L={matchedTemplate.skin_lab[0].toFixed(0)}</span>
                </div>
              )}
              <div className="comparison-item result-highlight">
                <h3>Result</h3>
                <img src={resultImage} alt="Result" />
              </div>
            </div>

            <a href={resultImage} download={`faceswap_${confirmedName}.jpg`} className="download-button mt-4">
              <Download size={18} /> Download Result
            </a>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;