import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import AuthService from '../services/AuthService';
import APIService from '../services/api';

const Discover = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedStyle, setSelectedStyle] = useState(null);
  const [showImageModal, setShowImageModal] = useState(false);
  const [modalImageUrl, setModalImageUrl] = useState('');
  const navigate = useNavigate();

  // Try Hairstyle Modal States
  const [showTryModal, setShowTryModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [overlayResult, setOverlayResult] = useState(null);
  const [overlayError, setOverlayError] = useState('');
  const [abortController, setAbortController] = useState(null);
  
  // Result Modal States
  const [showResultModal, setShowResultModal] = useState(false);
  
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Static hairstyle data organized by categories with real descriptions and images
  const hairstyleCategories = {
    natural: {
      name: "Natural Styles",
      icon: "🌿",
      styles: [
        {
          id: 1,
          name: "Natural Afro",
          category: "natural",
          gender: "male",
          length: "Medium to Long",
          maintenance: "High",
          theme: "Authentic & Bold",
          image: "/discover/natural-afro.jpg",
          description: "A hairstyle that embraces the natural texture and volume of coily or kinky hair, allowing it to grow outwards and upwards into a rounded shape. It's a statement of identity and requires significant moisture and care to prevent breakage.",
          features: ["Full volume", "Natural curl pattern", "Rounded shape", "Requires moisture"],
          suitableFor: ["Oval", "Round", "Square"],
          stylingTime: "10-20 minutes (daily moisturizing)",
          maintenanceLevel: "Regular deep conditioning (weekly), trims every 4-6 weeks",
          tags: ["Afro-textured", "Voluminous", "Coily", "Natural Hair Movement"]
        },
        {
          id: 2,
          name: "Surfer Hair (Beachy Waves)",
          category: "natural",
          gender: "male",
          length: "Medium to Long",
          maintenance: "Low",
          theme: "Relaxed & Carefree",
          image: "/discover/surfer-hair.jpg",
          description: "A low-maintenance, tousled hairstyle that looks wind-swept and sun-kissed. It's defined by natural-looking waves and texture, often enhanced with sea salt spray to mimic the effect of a day at the beach.",
          features: ["Tousled texture", "Natural waves", "Windswept look", "Often sun-kissed"],
          suitableFor: ["Oval", "Square", "Heart"],
          stylingTime: "5-10 minutes (air dry with spray)",
          maintenanceLevel: "Trim every 8-12 weeks",
          tags: ["Beachy", "Wavy", "Low-maintenance", "Tousled"]
        },
        {
          id: 3,
          name: "Wash-and-Go",
          category: "natural",
          gender: "female",
          length: "Short to Long (depends on curl type)",
          maintenance: "Medium",
          theme: "Effortless & Authentic",
          image: "/discover/Wash-and-Go.jpg",
          description: "A styling method for naturally curly or coily hair that involves cleansing, conditioning, and applying styling products (like gel or cream) to wet hair to define the natural curl pattern without heat or manipulation. The hair is then air-dried or diffused.",
          features: ["Defined natural curls", "No heat required", "Embraces texture", "Requires specific products"],
          suitableFor: ["All (for curly/coily hair)"],
          stylingTime: "20-40 minutes (plus drying time)",
          maintenanceLevel: "Wash day routine every 3-7 days",
          tags: ["Curly Girl Method", "Natural Curls", "Defined", "Heatless"]
        },
        {
          id: 4,
          name: "Long Layers",
          category: "natural",
          gender: "female",
          length: "Long",
          maintenance: "Low to Medium",
          theme: "Flowing & Versatile",
          image: "/discover/long-layers.jpg",
          description: "A simple, classic cut for long hair that adds movement, removes weight, and enhances natural texture (whether straight or wavy). The layers are typically soft and blended, allowing the hair to fall naturally with shape.",
          features: ["Adds movement", "Reduces bulk", "Enhances natural texture", "Versatile"],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "5-15 minutes (air dry or quick blow-dry)",
          maintenanceLevel: "Trim every 8-10 weeks",
          tags: ["Low-maintenance", "Versatile", "Flowing", "Blended"]
        }
      ]
    },
    casual: {
      name: "Casual Styles",
      icon: "☀️",
      styles: [
        {
          id: 5,
          name: "Messy Quiff",
          category: "casual",
          gender: "male",
          length: "Short to Medium",
          maintenance: "Medium",
          theme: "Effortless & Modern",
          image: "/discover/Messy-Quiff.jpg",
          description: "A relaxed version of the classic quiff, this style features volume and height at the front, but with a deliberately tousled and textured finish. The sides are typically shorter, and the top is styled loosely with fingers rather than a comb.",
          features: ["Textured volume", "Tousled top", "Short sides", "Effortless look"],
          suitableFor: ["Oval", "Round", "Square"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-5 weeks",
          tags: ["Textured", "Voluminous", "Relaxed", "Modern"]
        },
        {
          id: 6,
          name: "Buzz Cut",
          category: "casual",
          gender: "male",
          length: "Very Short",
          maintenance: "Low (styling) / High (upkeep)",
          theme: "Minimalist & Sharp",
          image: "/discover/Buzz-Cut.jpg",
          description: "A very short hairstyle where the hair is clipped close to the head using clippers. It's a no-fuss, masculine style that is extremely easy to style (it requires none) but needs frequent trims to maintain the clean look.",
          features: ["Extremely short", "Uniform length", "No styling needed", "Highlights facial features"],
          suitableFor: ["Oval", "Square", "Rectangular"],
          stylingTime: "0 minutes",
          maintenanceLevel: "Trim every 2-3 weeks",
          tags: ["Minimalist", "Low-maintenance", "Military", "Sharp"]
        },
        {
          id: 7,
          name: "Messy Bun",
          category: "casual",
          gender: "female",
          length: "Medium to Long",
          maintenance: "Low",
          theme: "Effortless & Practical",
          image: "/discover/Messy-Bun.jpg",
          description: "A popular and quick updo where the hair is gathered into a bun, but with a deliberately loose, undone, and textured finish. Strands are often left out to frame the face, making it a go-to for a relaxed, everyday look.",
          features: ["Tousled texture", "Quick updo", "Effortless", "Face-framing strands"],
          suitableFor: ["All"],
          stylingTime: "2-5 minutes",
          maintenanceLevel: "As needed",
          tags: ["Updo", "Relaxed", "Quick-style", "Undone"]
        },
        {
          id: 8,
          name: "Shoulder-Length Shag",
          category: "casual",
          gender: "female",
          length: "Medium",
          maintenance: "Low to Medium",
          theme: "Retro & Textured",
          image: "/discover/Shoulder-Length-Shag.jpg",
          description: "A modern take on the '70s shag, this cut features heavy layers, lots of texture, and often a fringe (like curtain bangs). It's designed to enhance natural waves and create a rock-and-roll, lived-in vibe with minimal effort.",
          features: ["Heavy layers", "Choppy texture", "Volume at the crown", "Often includes fringe"],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "5-15 minutes (scrunch with spray)",
          maintenanceLevel: "Trim every 6-8 weeks",
          tags: ["Layered", "Textured", "Retro", "Beachy"]
        }
      ]
    },
    classic: {
      name: "Classic Styles",
      icon: "👑",
      styles: [
        {
          id: 9,
          name: "Side Part",
          category: "classic",
          gender: "male",
          length: "Short to Medium",
          maintenance: "Medium",
          theme: "Timeless & Professional",
          image: "/discover/Side-Part.jpg",
          description: "A timeless men's hairstyle defined by a neat part on one side of the head. The hair on top has length and is combed over, while the sides are tapered or faded. It's a clean, polished look suitable for any occasion.",
          features: ["Defined part", "Tapered sides", "Combed-over top", "Polished finish"],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Professional", "Timeless", "Vintage", "Groomed"]
        },
        {
          id: 10,
          name: "Pompadour",
          category: "classic",
          gender: "male",
          length: "Medium (on top)",
          maintenance: "High",
          theme: "Retro & Bold",
          image: "/discover/Pompadour.jpg",
          description: "An iconic hairstyle featuring short sides and a long top that is swept upwards and back from the forehead, creating significant volume (the 'pomp'). It requires blow-drying and pomade to hold its dramatic shape.",
          features: ["High volume at front", "Short sides", "Slicked back", "Statement look"],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Vintage", "Rockabilly", "Voluminous", "High-maintenance"]
        },
        {
          id: 11,
          name: "Classic Bob",
          category: "classic",
          gender: "female",
          length: "Short (chin-length)",
          maintenance: "Medium",
          theme: "Timeless & Chic",
          image: "/discover/Classic-Bob.jpg",
          description: "A timeless cut where the hair is typically cut straight around the head at about jaw-level, often with a fringe. The 'classic' bob is precise, polished, and can be worn straight and sleek or with a slight bend.",
          features: ["Chin-length", "Precise line", "Often with fringe", "Polished look"],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "10-15 minutes (for sleek look)",
          maintenanceLevel: "Trim every 4-6 weeks to maintain shape",
          tags: ["Chic", "Polished", "Geometric", "Timeless"]
        },
        {
          id: 12,
          name: "French Twist",
          category: "classic",
          gender: "female",
          length: "Medium to Long",
          maintenance: "Medium",
          theme: "Elegant & Sophisticated",
          image: "/discover/French-Twist.jpg",
          description: "A sophisticated updo where hair is gathered, twisted vertically, and pinned neatly against the back of the head. It creates a sleek, polished 'roll' that is a go-to style for formal events and professional settings.",
          features: ["Vertical roll", "Sleek and polished", "Formal updo", "Securely pinned"],
          suitableFor: ["All"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Requires practice and pins",
          tags: ["Updo", "Formal", "Elegant", "Timeless"]
        }
      ]
    },
    elegant: {
      name: "Elegant Styles",
      icon: "✨",
      styles: [
        {
          id: 13,
          name: "Slick Back",
          category: "elegant",
          gender: "male",
          length: "Medium (on top)",
          maintenance: "High",
          theme: "Sharp & Sophisticated",
          image: "/discover/Slick-Back.jpg",
          description: "A sharp, polished hairstyle where the hair on top is combed straight back from the forehead, lying flat against the head. It typically features shorter sides (an undercut or fade) and requires a high-shine pomade for a sleek, wet look.",
          features: ["Combed straight back", "High-shine finish", "Undercut or fade sides", "Polished"],
          suitableFor: ["Oval", "Square"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Formal", "Polished", "High-shine", "Sharp"]
        },
        {
          id: 14,
          name: "Taper Fade with Comb Over",
          category: "elegant",
          gender: "male",
          length: "Short to Medium",
          maintenance: "Medium",
          theme: "Modern & Refined",
          image: "/discover/Taper-Fade-with-Comb-Over.jpg",
          description: "A modern and clean hairstyle that combines two classic elements. The taper fade provides a gradual, clean blend on the sides and back, while the longer top is neatly combed to one side, creating a defined part.",
          features: ["Gradual taper fade", "Defined side part", "Neatly combed top", "Clean and sharp"],
          suitableFor: ["All"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Professional", "Sharp", "Modern-classic", "Faded"]
        },
        {
          id: 15,
          name: "Chignon",
          category: "elegant",
          gender: "female",
          length: "Medium to Long",
          maintenance: "Medium",
          theme: "Graceful & Timeless",
          image: "/discover/Chignon.jpg",
          description: "A classic and elegant updo, typically worn at the nape of the neck. The hair is gathered into a low ponytail, then looped, twisted, or tucked into a sleek, graceful knot. It's a popular choice for weddings and formal events.",
          features: ["Low bun at nape", "Sleek and smooth", "Graceful knot", "Formal updo"],
          suitableFor: ["All"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Requires pins and hairspray",
          tags: ["Formal", "Updo", "Bridal", "Sophisticated"]
        },
        {
          id: 16,
          name: "Classic Updo",
          category: "elegant",
          gender: "female",
          length: "Medium to Long",
          maintenance: "High",
          theme: "Formal & Ornate",
          image: "/discover/Classic-Updo.jpg",
          description: "A formal hairstyle where the hair is swept up and secured away from the face and neck. This can range from intricate twists, braids, and curls to a voluminous, structured bun. It's designed for special occasions and black-tie events.",
          features: ["Hair swept off neck", "Intricate design (twists, pins)", "Voluminous", "Formal"],
          suitableFor: ["All"],
          stylingTime: "30-60+ minutes (often professional)",
          maintenanceLevel: "Special occasion style",
          tags: ["Formal", "Black-tie", "Bridal", "Ornate"]
        }
      ]
    },
    glamorous: {
      name: "Glamorous Styles",
      icon: "💎",
      styles: [
        {
          id: 17,
          name: "Quiff with High Shine",
          category: "glamorous",
          gender: "male",
          length: "Medium (on top)",
          maintenance: "High",
          theme: "Dapper & Show-Stopping",
          image: "/discover/Quiff-with-High-Shine.jpg",
          description: "This is a statement-making quiff that focuses on both volume and a wet-look, high-shine finish. It's styled using a blow-dryer for maximum height and a strong-hold, glossy pomade to catch the light.",
          features: ["Maximum volume", "High-shine finish", "Shorter sides", "Statement look"],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["High-shine", "Voluminous", "Statement", "Red carpet"]
        },
        {
          id: 18,
          name: "Long Wavy Hair",
          category: "glamorous",
          gender: "male",
          length: "Long",
          maintenance: "Medium",
          theme: "Rugged & Romantic",
          image: "/discover/Long-Wavy-Hair.jpg",
          description: "Long, flowing hair on men, often with a natural wave or curl. When styled for a glamorous look, it's healthy, shiny, and intentionally styled (either defined waves or a 'hero' sweep back) rather than just unkempt. Think red-carpet movie star.",
          features: ["Shoulder-length or longer", "Natural waves enhanced", "Healthy shine", "Can be tied or worn down"],
          suitableFor: ["Oval", "Square", "Heart"],
          stylingTime: "10-20 minutes (for definition)",
          maintenanceLevel: "Regular conditioning; trims every 10-12 weeks",
          tags: ["Flowing", "Wavy", "Rugged", "Romantic"]
        },
        {
          id: 19,
          name: "Hollywood Waves",
          category: "glamorous",
          gender: "female",
          length: "Medium to Long",
          maintenance: "High",
          theme: "Vintage & Red Carpet",
          image: "/discover/hollywood-waves.jpg",
          description: "A classic red-carpet hairstyle characterized by soft, uniform, and highly polished waves. The hair is typically deep-parted to one side and cascades over one shoulder, with a high-gloss finish.",
          features: ["Uniform S-shaped waves", "High-shine", "Deep side part", "Polished and structured"],
          suitableFor: ["All"],
          stylingTime: "30-60 minutes",
          maintenanceLevel: "Special occasion style",
          tags: ["Red carpet", "Vintage", "Polished", "Wavy"]
        },
        {
          id: 20,
          name: "Voluminous Blowout",
          category: "glamorous",
          gender: "female",
          length: "Medium to Long",
          maintenance: "High",
          theme: "Bouncy & Luxe",
          image: "/discover/blowout-hairstyle.webp",
          description: "A salon-quality blowout designed to create maximum volume, body, and movement. It involves using a round brush and blow-dryer to lift the roots and create soft, bouncy, shiny hair that looks full and healthy.",
          features: ["Maximum volume", "Bouncy movement", "High-shine", "Smooth finish"],
          suitableFor: ["All"],
          stylingTime: "20-45 minutes",
          maintenanceLevel: "Requires heat styling",
          tags: ["Bouncy", "Voluminous", "Luxe", "High-shine"]
        }
      ]
    },
    trendy: {
      name: "Trendy Styles",
      icon: "🔥",
      styles: [
        {
          id: 21,
          name: "Textured Crop (French Crop)",
          category: "trendy",
          gender: "male",
          length: "Short",
          maintenance: "Low to Medium",
          theme: "Modern & Edgy",
          image: "/discover/Textured-Crop.jpg",
          description: "A very popular modern cut featuring a short, textured top with a distinct fringe, contrasted by faded or undercut sides. The top is styled forward to create a messy, textured look. It's low-maintenance and stylish.",
          features: ["Textured top", "Short fringe (bangs)", "High fade or undercut", "Easy to style"],
          suitableFor: ["Oval", "Square", "Diamond"],
          stylingTime: "2-5 minutes (with matte clay or paste)",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Faded", "Textured", "Fringe", "Contemporary"]
        },
        {
          id: 22,
          name: "Modern Mullet",
          category: "trendy",
          gender: "male",
          length: "Short (sides) / Long (back)",
          maintenance: "Medium",
          theme: "Retro-Revival & Bold",
          image: "/discover/modern-mullet.webp",
          description: "A modern reinterpretation of the '80s classic. This version is more subtle, often featuring a taper fade on the sides, a textured top, and a less-dramatic, more blended length in the back. It's 'business in the front, party in the back' with a fashion-forward twist.",
          features: ["Short faded sides", "Longer back", "Textured top", "Retro-revival"],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 4-6 weeks to maintain shape",
          tags: ["Retro", "Edgy", "Faded", "Statement"]
        },
        {
          id: 23,
          name: "Wolf Cut",
          category: "trendy",
          gender: "female",
          length: "Medium to Long",
          maintenance: "Medium",
          theme: "Wild & Edgy",
          image: "/discover/Wolf-Cut.jpg",
          description: "A viral hybrid of a shag and a mullet. It features short, choppy layers on top for volume and longer, thinned-out layers in the back. It's defined by its wild texture and is often paired with curtain bangs.",
          features: ["Shag-mullet hybrid", "Heavy, choppy layers", "Volume at the crown", "Untamed texture"],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "10-15 minutes (with texturizing spray)",
          maintenanceLevel: "Trim every 6-8 weeks",
          tags: ["Viral", "Layered", "Edgy", "Textured"]
        },
        {
          id: 24,
          name: "Bixie Cut",
          category: "trendy",
          gender: "female",
          length: "Short",
          maintenance: "Low to Medium",
          theme: "Playful & Versatile",
          image: "/discover/Bixie-Cut.jpg",
          description: "A hybrid cut that blends the length and shape of a short bob with the layers and texture of a pixie cut. It's longer than a pixie but shorter than a bob, offering a soft, versatile, and low-maintenance short style.",
          features: ["Bob-pixie hybrid", "Soft, feathered layers", "Textured", "Low-maintenance"],
          suitableFor: ["Oval", "Heart", "Round"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 4-6 weeks",
          tags: ["Short hair", "Hybrid", "Layered", "Versatile"]
        }
      ]
    }
  };

  const categories = [
    { key: 'all', name: 'All Styles', icon: '🎨' },
    { key: 'natural', name: 'Natural', icon: '🌿' },
    { key: 'casual', name: 'Casual', icon: '☀️' },
    { key: 'classic', name: 'Classic', icon: '👑' },
    { key: 'elegant', name: 'Elegant', icon: '✨' },
    { key: 'glamorous', name: 'Glamorous', icon: '💎' },
    { key: 'trendy', name: 'Trendy', icon: '🔥' }
  ];

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error('Authentication check failed:', error);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  // Camera functions
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      setStream(mediaStream);
      setShowCamera(true);
      
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      }, 100);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Could not access camera. Please make sure you have granted camera permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setShowCamera(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        stopCamera();
      }, 'image/jpeg', 0.95);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      alert('Please select a valid image file');
    }
  };

  const handleTryStyle = () => {
    setShowTryModal(true);
    setSelectedFile(null);
    setPreviewUrl(null);
    setOverlayResult(null);
    setOverlayError('');
  };

  const closeTryModal = () => {
    setShowTryModal(false);
    setSelectedFile(null);
    setPreviewUrl(null);
    setOverlayResult(null);
    setOverlayError('');
    setIsProcessing(false);
    stopCamera();
  };

  const closeResultModal = () => {
    setShowResultModal(false);
    setOverlayResult(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    
    // Cancel any ongoing overlay generation
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
  };

  const handleGenerateOverlay = async () => {
    if (!selectedFile || !selectedStyle) {
      alert('Please select an image first');
      return;
    }

    setIsProcessing(true);
    setOverlayError('');
    setOverlayResult(null);

    // Create new AbortController for this request
    const controller = new AbortController();
    setAbortController(controller);

    try {
      // Step 1: Upload image
      const uploadResponse = await APIService.uploadImage(selectedFile);
      
      if (!uploadResponse.face_detected) {
        setOverlayError('No face detected in the image. Please try another photo.');
        setIsProcessing(false);
        return;
      }

      // Step 2: Search for hairstyle by name in the database
      // Since static IDs don't match DB IDs, we need to search by name
      // Backend expects 'q' parameter for search query
      const searchResponse = await APIService.searchHairstyles({ 
        q: selectedStyle.name,
        per_page: 1 
      });
      
      if (!searchResponse.results || searchResponse.results.length === 0) {
        setOverlayError(`The hairstyle "${selectedStyle.name}" is not yet available in our database. Please try another style.`);
        setIsProcessing(false);
        return;
      }

      const dbHairstyle = searchResponse.results[0];

      // Step 3: Generate overlay with the correct database ID
      const overlayResponse = await APIService.generateOverlay(
        uploadResponse.image_id,
        dbHairstyle.id,
        'advanced',
        controller.signal
      );

      setOverlayResult(overlayResponse);
      setIsProcessing(false);
      setAbortController(null);
      
      // Close the try modal and open the result modal
      setShowTryModal(false);
      setShowResultModal(true);
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Overlay generation cancelled');
        setOverlayError('Overlay generation cancelled');
      } else {
        console.error('Failed to generate overlay:', error);
        setOverlayError(error.message || 'Failed to generate overlay. Please try again.');
      }
      setIsProcessing(false);
      setAbortController(null);
    }
  };

  const handleCancelOverlay = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
    setIsProcessing(false);
    setOverlayError('Overlay generation cancelled');
  };

  const resolveMediaUrl = (url) => {
    if (!url) return '';
    if (url.startsWith('http://') || url.startsWith('https://')) return url;
    const serverOrigin = APIService.baseURL.replace(/\/api\/?$/, '');
    return `${serverOrigin}${url}`;
  };

  // Add keyboard support for closing image modal
  useEffect(() => {
    const handleEscKey = (e) => {
      if (e.key === 'Escape' && showImageModal) {
        setShowImageModal(false);
      }
    };

    document.addEventListener('keydown', handleEscKey);
    return () => document.removeEventListener('keydown', handleEscKey);
  }, [showImageModal]);

  // Prevent background scrolling when modals are open
  useEffect(() => {
    if (selectedStyle || showImageModal || showResultModal) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [selectedStyle, showImageModal, showResultModal]);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate('/');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const getAllStyles = () => {
    return Object.values(hairstyleCategories).flatMap(category => category.styles);
  };

  const getFilteredStyles = () => {
    if (selectedCategory === 'all') {
      return getAllStyles();
    }
    return hairstyleCategories[selectedCategory]?.styles || [];
  };

  const openStyleDetails = (style) => {
    setSelectedStyle(style);
  };

  const closeStyleDetails = () => {
    setSelectedStyle(null);
  };

  const openImageModal = (imageUrl, e) => {
    e.stopPropagation(); // Prevent opening style details modal
    setModalImageUrl(imageUrl);
    setShowImageModal(true);
  };

  const closeImageModal = () => {
    setShowImageModal(false);
    setModalImageUrl('');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <>
      <Navbar 
        user={user} 
        onLogout={handleLogout}
        transparent={true}
      />
      <div className="min-h-screen bg-gray-900 pt-20 md:pt-24">
        {/* Header Section with Modern Dark Theme */}
        <div className="bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900 py-16 md:py-24 relative overflow-hidden">
          {/* Dark geometric pattern background */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-96 h-96">
              <div className="w-full h-full rounded-full border-2 border-blue-400 transform translate-x-48 -translate-y-48"></div>
            </div>
            <div className="absolute top-1/4 left-0 w-64 h-64">
              <div className="w-full h-full rounded-full border-2 border-purple-400 transform -translate-x-32"></div>
            </div>
            {/* Mesh pattern overlay */}
            <svg className="absolute inset-0 w-full h-full" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
                  <path d="M 60 0 L 0 0 0 60" fill="none" stroke="rgb(59, 130, 246)" strokeWidth="0.5" opacity="0.3"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
            </svg>
          </div>

          <div className="max-w-7xl mx-auto px-4 text-center relative z-10">
            <div className="mb-6">
              <span className="inline-block bg-blue-500/20 text-blue-300 px-4 py-2 rounded-full text-sm font-medium border border-blue-500/30 backdrop-blur-sm">
                Explore Our Collection
              </span>
            </div>
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-blue-100 to-purple-200 bg-clip-text text-transparent">
              Discover Your Perfect
              <span className="block text-blue-400">Hairstyle</span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Explore our curated collection of hairstyles across different categories. 
              Find inspiration for your next look!
            </p>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 py-12 md:py-16">
          {/* Category Filter */}
          <div className="mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-8">Browse by Category</h2>
            <div className="flex flex-wrap gap-4">
              {categories.map((category) => (
                <button
                  key={category.key}
                  onClick={() => setSelectedCategory(category.key)}
                  className={`flex items-center space-x-3 px-6 py-4 rounded-lg font-semibold transition-all duration-300 transform ${
                    selectedCategory === category.key
                      ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-purple-500/25 scale-105 border border-blue-500/30'
                      : 'bg-white/5 text-gray-300 hover:bg-white/10 hover:text-white backdrop-blur-sm border border-white/10 hover:border-white/20 hover:scale-105'
                  }`}
                >
                  <span className="text-2xl">{category.icon}</span>
                  <span>{category.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Styles Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {getFilteredStyles().map((style) => (
              <div
                key={style.id}
                className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-purple-500/40 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300 cursor-pointer group"
                onClick={() => openStyleDetails(style)}
              >
                {/* Style Image */}
                <div className="relative mb-6 overflow-hidden rounded-lg">
                  <div className="w-full h-64 bg-gradient-to-br from-purple-600/20 to-blue-600/20 flex items-center justify-center">
                    {style.image ? (
                      <>
                        <img 
                          src={style.image} 
                          alt={style.name}
                          className="w-full h-full object-cover object-center group-hover:scale-110 transition-transform duration-500"
                          style={{ objectPosition: 'center 20%' }}
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextElementSibling.style.display = 'flex';
                          }}
                        />
                        <div className="hidden text-5xl items-center justify-center w-full h-full bg-gradient-to-br from-purple-600/20 to-blue-600/20">💇‍♀️</div>
                      </>
                    ) : (
                      <div className="text-5xl">💇‍♀️</div>
                    )}
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                </div>

                {/* Style Info */}
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-white group-hover:text-blue-400 transition-colors duration-300">
                    {style.name}
                  </h3>
                  <p className="text-gray-300 text-sm leading-relaxed line-clamp-2">
                    {style.description}
                  </p>

                  {/* Style Attributes */}
                  <div className="flex flex-wrap gap-2">
                    <span className="bg-purple-500/20 text-purple-300 px-3 py-1 rounded-full text-xs font-medium border border-purple-500/30">
                      {style.length}
                    </span>
                    <span className="bg-blue-500/20 text-blue-300 px-3 py-1 rounded-full text-xs font-medium border border-blue-500/30">
                      {style.maintenance} Maintenance
                    </span>
                    <span className="bg-green-500/20 text-green-300 px-3 py-1 rounded-full text-xs font-medium border border-green-500/30">
                      {style.theme}
                    </span>
                  </div>

                  {/* View Details Button */}
                  <div className="pt-3">
                    <div className="text-blue-400 text-sm font-semibold group-hover:text-blue-300 flex items-center">
                      View Details
                      <svg className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Style Details Modal */}
        {selectedStyle && (
          <div 
            className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4"
            onClick={closeStyleDetails}
          >
            <div 
              className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-white/10 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/10">
                <h2 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">{selectedStyle.name}</h2>
                <button
                  onClick={closeStyleDetails}
                  className="text-gray-400 hover:text-white transition-colors duration-300 p-2 hover:bg-white/10 rounded-lg"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Modal Content */}
              <div className="p-6 space-y-6">
                {/* Style Image */}
                <div className="relative w-full h-80 bg-gradient-to-br from-purple-600/20 to-blue-600/20 rounded-xl flex items-center justify-center border border-white/10 overflow-hidden group/image">
                  {selectedStyle.image ? (
                    <>
                      <img 
                        src={selectedStyle.image} 
                        alt={selectedStyle.name}
                        className="w-full h-full object-cover object-center cursor-pointer hover:scale-105 transition-transform duration-300"
                        style={{ objectPosition: 'center 20%' }}
                        onClick={(e) => openImageModal(selectedStyle.image, e)}
                        onError={(e) => {
                          e.target.style.display = 'none';
                          e.target.nextElementSibling.style.display = 'flex';
                        }}
                      />
                      <div className="hidden text-7xl items-center justify-center w-full h-full bg-gradient-to-br from-purple-600/20 to-blue-600/20">💇‍♀️</div>
                      {/* Click to enlarge indicator */}
                      <div className="absolute inset-0 bg-black/0 group-hover/image:bg-black/20 transition-colors duration-300 flex items-center justify-center pointer-events-none">
                        <div className="bg-white/90 backdrop-blur-sm text-gray-900 px-4 py-2 rounded-lg font-medium opacity-0 group-hover/image:opacity-100 transition-opacity duration-300">
                          🔍 Click to view full size
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="text-7xl">💇‍♀️</div>
                  )}
                </div>

                {/* Description */}
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                  <h3 className="text-lg font-semibold text-blue-400 mb-3">Description</h3>
                  <p className="text-gray-300 leading-relaxed">{selectedStyle.description}</p>
                </div>

                {/* Key Features */}
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                  <h3 className="text-lg font-semibold text-blue-400 mb-4">Key Features</h3>
                  <ul className="space-y-3">
                    {selectedStyle.features.map((feature, index) => (
                      <li key={index} className="flex items-start text-gray-300">
                        <span className="text-blue-400 mr-3 text-lg">✓</span>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Style Details Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                      <h4 className="text-sm font-semibold text-purple-400 mb-2">Hair Length</h4>
                      <p className="text-white font-medium">{selectedStyle.length}</p>
                    </div>
                    <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                      <h4 className="text-sm font-semibold text-purple-400 mb-2">Styling Time</h4>
                      <p className="text-white font-medium">{selectedStyle.stylingTime}</p>
                    </div>
                    <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                      <h4 className="text-sm font-semibold text-purple-400 mb-2">Maintenance</h4>
                      <p className="text-white font-medium">{selectedStyle.maintenanceLevel}</p>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                      <h4 className="text-sm font-semibold text-purple-400 mb-2">Best for Face Shapes</h4>
                      <p className="text-white font-medium">{selectedStyle.suitableFor.join(', ')}</p>
                    </div>
                    <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10">
                      <h4 className="text-sm font-semibold text-purple-400 mb-2">Style Tags</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedStyle.tags.map((tag, index) => (
                          <span key={index} className="bg-purple-500/20 text-purple-300 px-3 py-1 rounded-full text-xs font-medium border border-purple-500/30">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-4 pt-4">
                  <button
                    onClick={handleTryStyle}
                    className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-4 px-6 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-purple-500/25 border border-blue-500/30"
                  >
                    Try This Style
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Image Modal - Full screen view */}
        {showImageModal && modalImageUrl && (
          <div 
            className="fixed inset-0 bg-black/95 backdrop-blur-sm z-[60] flex items-center justify-center p-4"
            onClick={closeImageModal}
          >
            <div className="relative max-w-5xl w-full">
              {/* Close button */}
              <button
                onClick={closeImageModal}
                className="absolute -top-12 right-0 text-white hover:text-gray-300 transition-colors duration-300 flex items-center gap-2"
              >
                <span className="text-sm">Press ESC or click outside to close</span>
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              
              {/* Image container */}
              <div 
                className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-4 border border-white/20 shadow-2xl"
                onClick={(e) => e.stopPropagation()}
              >
                <h3 className="text-xl font-semibold text-white mb-4 text-center">
                  Full Size View
                </h3>
                <div className="relative">
                  <img
                    src={modalImageUrl}
                    alt="Full size hairstyle"
                    className="w-full h-auto rounded-lg shadow-2xl"
                    style={{ maxHeight: '80vh', objectFit: 'contain' }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Try Hairstyle Modal */}
        {showTryModal && selectedStyle && (
          <div 
            className="fixed inset-0 bg-black/90 backdrop-blur-md z-[70] flex items-center justify-center p-4"
            onClick={closeTryModal}
          >
            <div 
              className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-white/10 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/10 sticky top-0 bg-gradient-to-br from-slate-800 to-slate-900 z-10">
                <h2 className="text-2xl font-bold text-white">Try {selectedStyle.name}</h2>
                <button
                  onClick={closeTryModal}
                  className="text-gray-400 hover:text-white transition-colors duration-300 p-2 hover:bg-white/10 rounded-lg"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Modal Content */}
              <div className="p-6 space-y-6">
                {!previewUrl && !showCamera && !overlayResult && (
                  <>
                    <div className="text-center mb-4 sm:mb-6 px-2">
                      <p className="text-gray-300 text-base sm:text-lg mb-2">Upload your photo to see how this style looks on you!</p>
                      <p className="text-gray-400 text-xs sm:text-sm">Choose a clear front-facing photo for best results</p>
                    </div>

                    {/* Upload Options */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4">
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="flex flex-col items-center justify-center p-6 sm:p-8 bg-white/5 hover:bg-white/10 border-2 border-dashed border-white/20 hover:border-blue-500/50 rounded-xl transition-all duration-300 group"
                      >
                        <svg className="w-12 h-12 sm:w-16 sm:h-16 text-blue-400 mb-3 sm:mb-4 group-hover:scale-110 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <span className="text-white font-semibold text-base sm:text-lg mb-1 sm:mb-2">Upload Photo</span>
                        <span className="text-gray-400 text-xs sm:text-sm">Choose from your device</span>
                      </button>

                      <button
                        onClick={startCamera}
                        className="flex flex-col items-center justify-center p-6 sm:p-8 bg-white/5 hover:bg-white/10 border-2 border-dashed border-white/20 hover:border-purple-500/50 rounded-xl transition-all duration-300 group"
                      >
                        <svg className="w-12 h-12 sm:w-16 sm:h-16 text-purple-400 mb-3 sm:mb-4 group-hover:scale-110 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <span className="text-white font-semibold text-base sm:text-lg mb-1 sm:mb-2">Take Photo</span>
                        <span className="text-gray-400 text-xs sm:text-sm">Use your camera</span>
                      </button>
                    </div>

                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </>
                )}

                {/* Camera View */}
                {showCamera && (
                  <div className="space-y-4">
                    <div className="relative bg-black rounded-xl overflow-hidden">
                      <div className="w-full max-h-[60vh] md:max-h-[500px] flex items-center justify-center">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          className="max-w-full max-h-[60vh] md:max-h-[500px] w-auto h-auto object-contain"
                        />
                      </div>
                    </div>
                    <div className="flex flex-col sm:flex-row gap-3 sm:gap-4">
                      <button
                        onClick={capturePhoto}
                        className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-3 px-4 sm:px-6 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 text-sm sm:text-base"
                      >
                        📸 Capture Photo
                      </button>
                      <button
                        onClick={stopCamera}
                        className="px-4 sm:px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-lg font-semibold transition-all duration-300 border border-white/20 text-sm sm:text-base"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {/* Preview and Generate */}
                {previewUrl && !overlayResult && (
                  <div className="space-y-4">
                    <div className="relative bg-black rounded-xl overflow-hidden">
                      <div className="w-full max-h-[60vh] md:max-h-[500px] flex items-center justify-center">
                        <img
                          src={previewUrl}
                          alt="Preview"
                          className="max-w-full max-h-[60vh] md:max-h-[500px] w-auto h-auto object-contain rounded-lg"
                        />
                      </div>
                    </div>

                    {overlayError && (
                      <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4">
                        <p className="text-red-300 text-sm">{overlayError}</p>
                      </div>
                    )}

                    <div className="flex flex-col sm:flex-row gap-3 sm:gap-4">
                      {!isProcessing ? (
                        <>
                          <button
                            onClick={handleGenerateOverlay}
                            className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-3 px-4 sm:px-6 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg text-sm sm:text-base"
                          >
                            ✨ Generate Hairstyle Preview
                          </button>
                          <button
                            onClick={() => {
                              setPreviewUrl(null);
                              setSelectedFile(null);
                              setOverlayError('');
                            }}
                            className="px-4 sm:px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-lg font-semibold transition-all duration-300 border border-white/20 text-sm sm:text-base"
                          >
                            Choose Different Photo
                          </button>
                        </>
                      ) : (
                        <button
                          onClick={handleCancelOverlay}
                          className="flex-1 bg-red-500/20 hover:bg-red-500/30 text-red-300 py-3 px-4 sm:px-6 rounded-lg font-semibold transition-all duration-300 border border-red-500/50 text-sm sm:text-base"
                        >
                          ⏹️ Cancel Generation
                        </button>
                      )}
                    </div>

                    {isProcessing && (
                      <div className="bg-blue-500/20 border border-blue-500/50 rounded-lg p-4 sm:p-6 text-center">
                        <div className="flex flex-col items-center space-y-3 sm:space-y-4">
                          <div className="animate-spin rounded-full h-10 w-10 sm:h-12 sm:w-12 border-4 border-blue-500 border-t-transparent"></div>
                          <div className="space-y-1 sm:space-y-2">
                            <p className="text-white font-semibold text-sm sm:text-base">Generating your hairstyle preview...</p>
                            <p className="text-gray-300 text-xs sm:text-sm">This may take a moment. Please wait.</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}


              </div>
            </div>

            {/* Hidden canvas for camera capture */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
        )}

        {/* Result Modal - Shows after overlay generation */}
        {showResultModal && overlayResult && selectedStyle && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn">
            <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-white/10">
              {/* Modal Header */}
              <div className="sticky top-0 bg-gradient-to-r from-gray-900 to-gray-800 border-b border-white/10 p-4 sm:p-6 z-10">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl sm:text-3xl font-bold text-white mb-1">
                      ✨ Your New Look!
                    </h2>
                    <p className="text-gray-400 text-sm sm:text-base">
                      {selectedStyle.name} Preview
                    </p>
                  </div>
                  <button
                    onClick={closeResultModal}
                    className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-white/10 rounded-lg"
                  >
                    <svg className="w-6 h-6 sm:w-8 sm:h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Modal Content */}
              <div className="p-4 sm:p-6 space-y-6">
                {/* Image Comparison */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                  {/* Original Photo */}
                  <div className="space-y-2 sm:space-y-3">
                    <h3 className="text-white font-semibold text-center text-base sm:text-lg">
                      Original Photo
                    </h3>
                    <div className="relative bg-black rounded-xl overflow-hidden shadow-lg h-[60vh] md:h-[500px] flex items-center justify-center">
                      <img
                        src={previewUrl}
                        alt="Original"
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </div>

                  {/* Styled Photo */}
                  <div className="space-y-2 sm:space-y-3">
                    <h3 className="text-white font-semibold text-center text-base sm:text-lg">
                      With {selectedStyle.name}
                    </h3>
                    <div className="relative bg-black rounded-xl overflow-hidden shadow-lg h-[60vh] md:h-[500px] flex items-center justify-center">
                      <img
                        src={resolveMediaUrl(overlayResult.overlay_url)}
                        alt="With hairstyle"
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </div>
                </div>

                {/* Info Box */}
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 sm:p-6">
                  <div className="flex items-start space-x-3">
                    <svg className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div className="flex-1">
                      <h4 className="text-white font-semibold mb-2 text-sm sm:text-base">About This Preview</h4>
                      <p className="text-gray-300 text-xs sm:text-sm leading-relaxed">
                        This is an AI-generated preview to help you visualize how <span className="font-semibold text-white">{selectedStyle.name}</span> might look on you. 
                        Actual results may vary based on your hair type, texture, and stylist expertise. 
                        We recommend consulting with a professional stylist for the best results.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-3 sm:gap-4">
                  <button
                    onClick={() => {
                      closeResultModal();
                      handleTryStyle();
                    }}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-3 px-4 sm:px-6 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg text-sm sm:text-base"
                  >
                    🔄 Try Another Photo
                  </button>
                  <button
                    onClick={closeResultModal}
                    className="px-4 sm:px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-lg font-semibold transition-all duration-300 border border-white/20 text-sm sm:text-base"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default Discover;
