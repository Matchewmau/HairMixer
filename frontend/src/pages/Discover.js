import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import AuthService from "../services/AuthService";
import APIService from "../services/api";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";
import Modal from "../components/ui/Modal";

const Discover = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedStyle, setSelectedStyle] = useState(null);
  const [showImageModal, setShowImageModal] = useState(false);
  const [modalImageUrl, setModalImageUrl] = useState("");
  const navigate = useNavigate();

  // Try Hairstyle Modal States
  const [showTryModal, setShowTryModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [overlayResult, setOverlayResult] = useState(null);
  const [overlayError, setOverlayError] = useState("");
  const [abortController, setAbortController] = useState(null);

  // Result Modal States
  const [showResultModal, setShowResultModal] = useState(false);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Static hairstyle data (kept same as original)
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
          description:
            "A hairstyle that embraces the natural texture and volume of coily or kinky hair, allowing it to grow outwards and upwards into a rounded shape. It's a statement of identity and requires significant moisture and care to prevent breakage.",
          features: [
            "Full volume",
            "Natural curl pattern",
            "Rounded shape",
            "Requires moisture",
          ],
          suitableFor: ["Oval", "Round", "Square"],
          stylingTime: "10-20 minutes (daily moisturizing)",
          maintenanceLevel:
            "Regular deep conditioning (weekly), trims every 4-6 weeks",
          tags: [
            "Afro-textured",
            "Voluminous",
            "Coily",
            "Natural Hair Movement",
          ],
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
          description:
            "A low-maintenance, tousled hairstyle that looks wind-swept and sun-kissed. It's defined by natural-looking waves and texture, often enhanced with sea salt spray to mimic the effect of a day at the beach.",
          features: [
            "Tousled texture",
            "Natural waves",
            "Windswept look",
            "Often sun-kissed",
          ],
          suitableFor: ["Oval", "Square", "Heart"],
          stylingTime: "5-10 minutes (air dry with spray)",
          maintenanceLevel: "Trim every 8-12 weeks",
          tags: ["Beachy", "Wavy", "Low-maintenance", "Tousled"],
        },
        {
          id: 3,
          name: "Wash-and-Go-Curls",
          category: "natural",
          gender: "female",
          length: "Short to Long (depends on curl type)",
          maintenance: "Medium",
          theme: "Effortless & Authentic",
          image: "/discover/Wash-and-Go.jpg",
          description:
            "A styling method for naturally curly or coily hair that involves cleansing, conditioning, and applying styling products (like gel or cream) to wet hair to define the natural curl pattern without heat or manipulation. The hair is then air-dried or diffused.",
          features: [
            "Defined natural curls",
            "No heat required",
            "Embraces texture",
            "Requires specific products",
          ],
          suitableFor: ["All (for curly/coily hair)"],
          stylingTime: "20-40 minutes (plus drying time)",
          maintenanceLevel: "Wash day routine every 3-7 days",
          tags: ["Curly Girl Method", "Natural Curls", "Defined", "Heatless"],
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
          description:
            "A simple, classic cut for long hair that adds movement, removes weight, and enhances natural texture (whether straight or wavy). The layers are typically soft and blended, allowing the hair to fall naturally with shape.",
          features: [
            "Adds movement",
            "Reduces bulk",
            "Enhances natural texture",
            "Versatile",
          ],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "5-15 minutes (air dry or quick blow-dry)",
          maintenanceLevel: "Trim every 8-10 weeks",
          tags: ["Low-maintenance", "Versatile", "Flowing", "Blended"],
        },
      ],
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
          description:
            "A relaxed version of the classic quiff, this style features volume and height at the front, but with a deliberately tousled and textured finish. The sides are typically shorter, and the top is styled loosely with fingers rather than a comb.",
          features: [
            "Textured volume",
            "Tousled top",
            "Short sides",
            "Effortless look",
          ],
          suitableFor: ["Oval", "Round", "Square"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-5 weeks",
          tags: ["Textured", "Voluminous", "Relaxed", "Modern"],
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
          description:
            "A very short hairstyle where the hair is clipped close to the head using clippers. It's a no-fuss, masculine style that is extremely easy to style (it requires none) but needs frequent trims to maintain the clean look.",
          features: [
            "Extremely short",
            "Uniform length",
            "No styling needed",
            "Highlights facial features",
          ],
          suitableFor: ["Oval", "Square", "Rectangular"],
          stylingTime: "0 minutes",
          maintenanceLevel: "Trim every 2-3 weeks",
          tags: ["Minimalist", "Low-maintenance", "Military", "Sharp"],
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
          description:
            "A popular and quick updo where the hair is gathered into a bun, but with a deliberately loose, undone, and textured finish. Strands are often left out to frame the face, making it a go-to for a relaxed, everyday look.",
          features: [
            "Tousled texture",
            "Quick updo",
            "Effortless",
            "Face-framing strands",
          ],
          suitableFor: ["All"],
          stylingTime: "2-5 minutes",
          maintenanceLevel: "As needed",
          tags: ["Updo", "Relaxed", "Quick-style", "Undone"],
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
          description:
            "A modern take on the '70s shag, this cut features heavy layers, lots of texture, and often a fringe (like curtain bangs). It's designed to enhance natural waves and create a rock-and-roll, lived-in vibe with minimal effort.",
          features: [
            "Heavy layers",
            "Choppy texture",
            "Volume at the crown",
            "Often includes fringe",
          ],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "5-15 minutes (scrunch with spray)",
          maintenanceLevel: "Trim every 6-8 weeks",
          tags: ["Layered", "Textured", "Retro", "Beachy"],
        },
      ],
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
          description:
            "A timeless men's hairstyle defined by a neat part on one side of the head. The hair on top has length and is combed over, while the sides are tapered or faded. It's a clean, polished look suitable for any occasion.",
          features: [
            "Defined part",
            "Tapered sides",
            "Combed-over top",
            "Polished finish",
          ],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Professional", "Timeless", "Vintage", "Groomed"],
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
          description:
            "An iconic hairstyle featuring short sides and a long top that is swept upwards and back from the forehead, creating significant volume (the 'pomp'). It requires blow-drying and pomade to hold its dramatic shape.",
          features: [
            "High volume at front",
            "Short sides",
            "Slicked back",
            "Statement look",
          ],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Vintage", "Rockabilly", "Voluminous", "High-maintenance"],
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
          description:
            "A timeless cut where the hair is typically cut straight around the head at about jaw-level, often with a fringe. The 'classic' bob is precise, polished, and can be worn straight and sleek or with a slight bend.",
          features: [
            "Chin-length",
            "Precise line",
            "Often with fringe",
            "Polished look",
          ],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "10-15 minutes (for sleek look)",
          maintenanceLevel: "Trim every 4-6 weeks to maintain shape",
          tags: ["Chic", "Polished", "Geometric", "Timeless"],
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
          description:
            "A sophisticated updo where hair is gathered, twisted vertically, and pinned neatly against the back of the head. It creates a sleek, polished 'roll' that is a go-to style for formal events and professional settings.",
          features: [
            "Vertical roll",
            "Sleek and polished",
            "Formal updo",
            "Securely pinned",
          ],
          suitableFor: ["All"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Requires practice and pins",
          tags: ["Updo", "Formal", "Elegant", "Timeless"],
        },
      ],
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
          description:
            "A sharp, polished hairstyle where the hair on top is combed straight back from the forehead, lying flat against the head. It typically features shorter sides (an undercut or fade) and requires a high-shine pomade for a sleek, wet look.",
          features: [
            "Combed straight back",
            "High-shine finish",
            "Undercut or fade sides",
            "Polished",
          ],
          suitableFor: ["Oval", "Square"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Formal", "Polished", "High-shine", "Sharp"],
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
          description:
            "A modern and clean hairstyle that combines two classic elements. The taper fade provides a gradual, clean blend on the sides and back, while the longer top is neatly combed to one side, creating a defined part.",
          features: [
            "Gradual taper fade",
            "Defined side part",
            "Neatly combed top",
            "Clean and sharp",
          ],
          suitableFor: ["All"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Professional", "Sharp", "Modern-classic", "Faded"],
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
          description:
            "A classic and elegant updo, typically worn at the nape of the neck. The hair is gathered into a low ponytail, then looped, twisted, or tucked into a sleek, graceful knot. It's a popular choice for weddings and formal events.",
          features: [
            "Low bun at nape",
            "Sleek and smooth",
            "Graceful knot",
            "Formal updo",
          ],
          suitableFor: ["All"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Requires pins and hairspray",
          tags: ["Formal", "Updo", "Bridal", "Sophisticated"],
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
          description:
            "A formal hairstyle where the hair is swept up and secured away from the face and neck. This can range from intricate twists, braids, and curls to a voluminous, structured bun. It's designed for special occasions and black-tie events.",
          features: [
            "Hair swept off neck",
            "Intricate design (twists, pins)",
            "Voluminous",
            "Formal",
          ],
          suitableFor: ["All"],
          stylingTime: "30-60+ minutes (often professional)",
          maintenanceLevel: "Special occasion style",
          tags: ["Formal", "Black-tie", "Bridal", "Ornate"],
        },
      ],
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
          description:
            "This is a statement-making quiff that focuses on both volume and a wet-look, high-shine finish. It's styled using a blow-dryer for maximum height and a strong-hold, glossy pomade to catch the light.",
          features: [
            "Maximum volume",
            "High-shine finish",
            "Shorter sides",
            "Statement look",
          ],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["High-shine", "Voluminous", "Statement", "Red carpet"],
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
          description:
            "Long, flowing hair on men, often with a natural wave or curl. When styled for a glamorous look, it's healthy, shiny, and intentionally styled (either defined waves or a 'hero' sweep back) rather than just unkempt. Think red-carpet movie star.",
          features: [
            "Shoulder-length or longer",
            "Natural waves enhanced",
            "Healthy shine",
            "Can be tied or worn down",
          ],
          suitableFor: ["Oval", "Square", "Heart"],
          stylingTime: "10-20 minutes (for definition)",
          maintenanceLevel: "Regular conditioning; trims every 10-12 weeks",
          tags: ["Flowing", "Wavy", "Rugged", "Romantic"],
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
          description:
            "A classic red-carpet hairstyle characterized by soft, uniform, and highly polished waves. The hair is typically deep-parted to one side and cascades over one shoulder, with a high-gloss finish.",
          features: [
            "Uniform S-shaped waves",
            "High-shine",
            "Deep side part",
            "Polished and structured",
          ],
          suitableFor: ["All"],
          stylingTime: "30-60 minutes",
          maintenanceLevel: "Special occasion style",
          tags: ["Red carpet", "Vintage", "Polished", "Wavy"],
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
          description:
            "A salon-quality blowout designed to create maximum volume, body, and movement. It involves using a round brush and blow-dryer to lift the roots and create soft, bouncy, shiny hair that looks full and healthy.",
          features: [
            "Maximum volume",
            "Bouncy movement",
            "High-shine",
            "Smooth finish",
          ],
          suitableFor: ["All"],
          stylingTime: "20-45 minutes",
          maintenanceLevel: "Requires heat styling",
          tags: ["Bouncy", "Voluminous", "Luxe", "High-shine"],
        },
      ],
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
          description:
            "A very popular modern cut featuring a short, textured top with a distinct fringe, contrasted by faded or undercut sides. The top is styled forward to create a messy, textured look. It's low-maintenance and stylish.",
          features: [
            "Textured top",
            "Short fringe (bangs)",
            "High fade or undercut",
            "Easy to style",
          ],
          suitableFor: ["Oval", "Square", "Diamond"],
          stylingTime: "2-5 minutes (with matte clay or paste)",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Faded", "Textured", "Fringe", "Contemporary"],
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
          description:
            "A modern reinterpretation of the '80s classic. This version is more subtle, often featuring a taper fade on the sides, a textured top, and a less-dramatic, more blended length in the back. It's 'business in the front, party in the back' with a fashion-forward twist.",
          features: [
            "Short faded sides",
            "Longer back",
            "Textured top",
            "Retro-revival",
          ],
          suitableFor: ["Oval", "Square", "Round"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 4-6 weeks to maintain shape",
          tags: ["Retro", "Edgy", "Faded", "Statement"],
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
          description:
            "A viral hybrid of a shag and a mullet. It features short, choppy layers on top for volume and longer, thinned-out layers in the back. It's defined by its wild texture and is often paired with curtain bangs.",
          features: [
            "Shag-mullet hybrid",
            "Heavy, choppy layers",
            "Volume at the crown",
            "Untamed texture",
          ],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "10-15 minutes (with texturizing spray)",
          maintenanceLevel: "Trim every 6-8 weeks",
          tags: ["Viral", "Layered", "Edgy", "Textured"],
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
          description:
            "A hybrid cut that blends the length and shape of a short bob with the layers and texture of a pixie cut. It's longer than a pixie but shorter than a bob, offering a soft, versatile, and low-maintenance short style.",
          features: [
            "Bob-pixie hybrid",
            "Soft, feathered layers",
            "Textured",
            "Low-maintenance",
          ],
          suitableFor: ["Oval", "Heart", "Round"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 4-6 weeks",
          tags: ["Short hair", "Hybrid", "Layered", "Versatile"],
        },
      ],
    },
  };

  const categories = [
    { key: "all", name: "All Styles", icon: "🎨" },
    { key: "natural", name: "Natural", icon: "🌿" },
    { key: "casual", name: "Casual", icon: "☀️" },
    { key: "classic", name: "Classic", icon: "👑" },
    { key: "elegant", name: "Elegant", icon: "✨" },
    { key: "glamorous", name: "Glamorous", icon: "💎" },
    { key: "trendy", name: "Trendy", icon: "🔥" },
  ];

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error("Authentication check failed:", error);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      setStream(mediaStream);
      setShowCamera(true);

      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      }, 100);
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert(
        "Could not access camera. Please make sure you have granted camera permissions."
      );
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
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

      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          const file = new File([blob], "camera-photo.jpg", {
            type: "image/jpeg",
          });
          setSelectedFile(file);
          setPreviewUrl(URL.createObjectURL(file));
          stopCamera();
        },
        "image/jpeg",
        0.95
      );
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      alert("Please select a valid image file");
    }
  };

  const handleTryStyle = () => {
    setShowTryModal(true);
    setSelectedFile(null);
    setPreviewUrl(null);
    setOverlayResult(null);
    setOverlayError("");
  };

  const closeTryModal = () => {
    setShowTryModal(false);
    setSelectedFile(null);
    setPreviewUrl(null);
    setOverlayResult(null);
    setOverlayError("");
    setIsProcessing(false);
    stopCamera();
  };

  const closeResultModal = () => {
    setShowResultModal(false);
    setOverlayResult(null);
    setSelectedFile(null);
    setPreviewUrl(null);

    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
  };

  const handleGenerateOverlay = async () => {
    if (!selectedFile || !selectedStyle) {
      alert("Please select an image first");
      return;
    }

    setIsProcessing(true);
    setOverlayError("");
    setOverlayResult(null);

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const uploadResponse = await APIService.uploadImage(selectedFile);

      if (!uploadResponse.face_detected) {
        setOverlayError(
          "No face detected in the image. Please try another photo."
        );
        setIsProcessing(false);
        return;
      }

      const searchResponse = await APIService.searchHairstyles({
        q: selectedStyle.name,
        per_page: 1,
      });

      if (!searchResponse.results || searchResponse.results.length === 0) {
        setOverlayError(
          `The hairstyle "${selectedStyle.name}" is not yet available in our database. Please try another style.`
        );
        setIsProcessing(false);
        return;
      }

      const dbHairstyle = searchResponse.results[0];

      const overlayResponse = await APIService.generateOverlay(
        uploadResponse.image_id,
        dbHairstyle.id,
        "advanced",
        controller.signal
      );

      setOverlayResult(overlayResponse);
      setIsProcessing(false);
      setAbortController(null);

      setShowTryModal(false);
      setShowResultModal(true);
    } catch (error) {
      if (error.name === "AbortError") {
        console.log("Overlay generation cancelled");
        setOverlayError("Overlay generation cancelled");
      } else {
        console.error("Failed to generate overlay:", error);
        setOverlayError(
          error.message || "Failed to generate overlay. Please try again."
        );
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
    setOverlayError("Overlay generation cancelled");
  };

  const resolveMediaUrl = (url) => {
    if (!url) return "";
    if (url.startsWith("http://") || url.startsWith("https://")) return url;
    const serverOrigin = APIService.baseURL.replace(/\/api\/?$/, "");
    return `${serverOrigin}${url}`;
  };

  const handleSaveImage = async () => {
    if (!overlayResult?.overlay_url) return;

    try {
      const imageUrl = resolveMediaUrl(overlayResult.overlay_url);
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `hairmixer-makeover-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error downloading image:", error);
      alert("Failed to download image. Please try again.");
    }
  };

  useEffect(() => {
    const handleEscKey = (e) => {
      if (e.key === "Escape" && showImageModal) {
        setShowImageModal(false);
      }
    };

    document.addEventListener("keydown", handleEscKey);
    return () => document.removeEventListener("keydown", handleEscKey);
  }, [showImageModal]);

  useEffect(() => {
    if (selectedStyle || showImageModal || showResultModal) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }

    return () => {
      document.body.style.overflow = "unset";
    };
  }, [selectedStyle, showImageModal, showResultModal]);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const getAllStyles = () => {
    return Object.values(hairstyleCategories).flatMap(
      (category) => category.styles
    );
  };

  const getFilteredStyles = () => {
    if (selectedCategory === "all") {
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
    e.stopPropagation();
    setModalImageUrl(imageUrl);
    setShowImageModal(true);
  };

  const closeImageModal = () => {
    setShowImageModal(false);
    setModalImageUrl("");
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background animate-fade-in">
      <Navbar user={user} onLogout={handleLogout} transparent={true} />

      <div className="pt-24 pb-12">
        {/* Header Section */}
        <div className="bg-gradient-to-br from-background via-surface to-primary/20 py-14 md:py-14 relative overflow-hidden">
          {/* Background Elements */}
          <div className="absolute top-0 right-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl opacity-30 pointer-events-none"></div>
          <div className="absolute bottom-0 left-0 w-96 h-96 bg-secondary/10 rounded-full blur-3xl opacity-30 pointer-events-none"></div>

          <div className="max-w-7xl mx-auto px-4 text-center relative z-10">
            <div className="mb-6">
              <span className="inline-block bg-primary/20 text-primary-foreground px-4 py-2 rounded-full text-sm font-medium border border-primary/30 backdrop-blur-sm animate-slide-up">
                Explore Our Collection
              </span>
            </div>
            <h1
              className="text-4xl md:text-6xl lg:text-7xl font-heading font-bold mb-6 text-white animate-slide-up"
              style={{ animationDelay: "0.1s" }}
            >
              Discover Your Perfect
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">
                Hairstyle
              </span>
            </h1>
            <p
              className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed animate-slide-up"
              style={{ animationDelay: "0.2s" }}
            >
              Explore our curated collection of hairstyles across different
              categories. Find inspiration for your next look!
            </p>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 py-12 md:py-16">
          {/* Category Filter */}
          <div className="mb-16">
            <h2 className="text-3xl md:text-4xl font-heading font-bold text-white mb-8">
              Browse by Category
            </h2>
            <div className="flex flex-wrap gap-4">
              {categories.map((category) => (
                <Button
                  key={category.key}
                  onClick={() => setSelectedCategory(category.key)}
                  variant={
                    selectedCategory === category.key ? "primary" : "outline"
                  }
                  className={`flex items-center space-x-3 px-6 py-4 rounded-lg font-semibold transition-all duration-300 ${
                    selectedCategory === category.key
                      ? "scale-105 shadow-lg shadow-primary/25"
                      : "hover:bg-surface/50"
                  }`}
                >
                  <span className="text-2xl">{category.icon}</span>
                  <span>{category.name}</span>
                </Button>
              ))}
            </div>
          </div>

          {/* Styles Grid */}
          <div
            key={selectedCategory}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8 animate-slide-up"
          >
            {getFilteredStyles().map((style) => (
              <Card
                key={style.id}
                hover
                className="group cursor-pointer overflow-hidden flex flex-col h-full"
                onClick={() => openStyleDetails(style)}
              >
                <div className="relative h-64 overflow-hidden">
                  <img
                    src={style.image}
                    alt={style.name}
                    className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-80"></div>
                  <div className="absolute bottom-0 left-0 right-0 p-6">
                    <h3 className="text-xl font-heading font-bold text-white mb-1 group-hover:text-primary transition-colors">
                      {style.name}
                    </h3>
                    <div className="flex items-center space-x-2 text-sm text-gray-300">
                      <span className="capitalize">{style.gender}</span>
                      <span>•</span>
                      <span>{style.length}</span>
                    </div>
                  </div>
                </div>

                <div className="p-6 flex-1 flex flex-col">
                  <p className="text-gray-400 text-sm line-clamp-3 mb-4 flex-1">
                    {style.description}
                  </p>

                  <div className="flex flex-wrap gap-2 mb-4">
                    {style.tags.slice(0, 3).map((tag, idx) => (
                      <span
                        key={idx}
                        className="text-xs bg-surface/50 text-gray-300 px-2 py-1 rounded border border-white/5"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full group-hover:bg-primary group-hover:text-white group-hover:border-primary transition-colors"
                  >
                    View Details
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* Style Details Modal */}
      <Modal
        isOpen={!!selectedStyle}
        onClose={closeStyleDetails}
        title={selectedStyle?.name || "Hairstyle Details"}
        size="lg"
      >
        {selectedStyle && (
          <div className="space-y-8">
            <div
              className="relative h-80 rounded-xl overflow-hidden group cursor-pointer"
              onClick={(e) => openImageModal(selectedStyle.image, e)}
            >
              <img
                src={selectedStyle.image}
                alt={selectedStyle.name}
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
                <span className="bg-black/60 text-white px-4 py-2 rounded-full backdrop-blur-sm">
                  View Full Image
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-heading font-bold text-white mb-4 border-b border-white/10 pb-2">
                  Description
                </h3>
                <p className="text-gray-300 leading-relaxed mb-6">
                  {selectedStyle.description}
                </p>

                <h3 className="text-lg font-heading font-bold text-white mb-4 border-b border-white/10 pb-2">
                  Key Features
                </h3>
                <ul className="space-y-2 mb-6">
                  {selectedStyle.features.map((feature, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-2 text-gray-300"
                    >
                      <span className="text-primary mt-1">✓</span>
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="space-y-6">
                <div className="bg-surface/50 rounded-xl p-6 border border-white/5">
                  <h3 className="text-lg font-heading font-bold text-white mb-4">
                    Style Info
                  </h3>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Category:</span>{" "}
                      <span className="text-white capitalize">
                        {selectedStyle.category}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Length:</span>{" "}
                      <span className="text-white">{selectedStyle.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Maintenance:</span>{" "}
                      <span className="text-white">
                        {selectedStyle.maintenance}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Styling Time:</span>{" "}
                      <span className="text-white">
                        {selectedStyle.stylingTime}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-surface/50 rounded-xl p-6 border border-white/5">
                  <h3 className="text-lg font-heading font-bold text-white mb-4">
                    Best Suited For
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedStyle.suitableFor.map((face, idx) => (
                      <span
                        key={idx}
                        className="bg-primary/20 text-primary px-3 py-1 rounded-full text-sm font-medium border border-primary/30"
                      >
                        {face}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-4 pt-6 border-t border-white/10">
              <Button onClick={closeStyleDetails} variant="ghost">
                Close
              </Button>
              <Button
                onClick={handleTryStyle}
                variant="primary"
                className="shadow-lg shadow-primary/20"
              >
                Try This Hairstyle
              </Button>
            </div>
          </div>
        )}
      </Modal>

      {/* Try Hairstyle Modal */}
      <Modal
        isOpen={showTryModal}
        onClose={closeTryModal}
        title={`Try on: ${selectedStyle?.name}`}
        size="lg"
      >
        <div className="space-y-6">
          {!previewUrl ? (
            <div className="space-y-6">
              <p className="text-gray-300 text-center">
                Upload a photo or use your camera to see how this hairstyle
                looks on you!
              </p>

              {showCamera ? (
                <div className="relative rounded-xl overflow-hidden bg-black aspect-video">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-4">
                    <Button onClick={capturePhoto} variant="primary">
                      Capture
                    </Button>
                    <Button onClick={stopCamera} variant="secondary">
                      Cancel
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <Card
                    className="p-8 flex flex-col items-center justify-center gap-4 cursor-pointer hover:border-primary/50 transition-colors border-dashed"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <span className="text-4xl">📁</span>
                    <span className="font-medium text-white">Upload Photo</span>
                  </Card>
                  <Card
                    className="p-8 flex flex-col items-center justify-center gap-4 cursor-pointer hover:border-primary/50 transition-colors border-dashed"
                    onClick={startCamera}
                  >
                    <span className="text-4xl">📸</span>
                    <span className="font-medium text-white">Use Camera</span>
                  </Card>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          ) : (
            <div className="space-y-6">
              <div className="relative rounded-xl overflow-hidden aspect-[3/4] max-h-[50vh] mx-auto">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full h-full object-cover"
                />
                {isProcessing && (
                  <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mb-4"></div>
                    <p className="text-white font-medium">
                      Generating your new look...
                    </p>
                    <Button
                      onClick={handleCancelOverlay}
                      variant="ghost"
                      className="mt-4 text-red-400 hover:text-red-300"
                    >
                      Cancel
                    </Button>
                  </div>
                )}
              </div>

              {overlayError && (
                <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-lg text-center">
                  {overlayError}
                </div>
              )}

              <div className="flex justify-center gap-4">
                <Button
                  onClick={() => setPreviewUrl(null)}
                  variant="secondary"
                  disabled={isProcessing}
                >
                  Change Photo
                </Button>
                <Button
                  onClick={handleGenerateOverlay}
                  variant="primary"
                  disabled={isProcessing}
                >
                  Generate Look
                </Button>
              </div>
            </div>
          )}
        </div>
        <canvas ref={canvasRef} className="hidden" />
      </Modal>

      {/* Result Modal */}
      <Modal isOpen={showResultModal} title="Your New Look" size="lg">
        {overlayResult && (
          <div className="space-y-6">
            {/* Before & After Comparison */}
            <div className="bg-surface/50 rounded-2xl p-4 border border-white/5">
              <h3 className="text-lg font-heading font-bold text-white mb-4 flex items-center gap-2">
                <span className="text-xl">✨</span> Transformation Result
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="relative aspect-[3/4] rounded-xl overflow-hidden border border-white/10 bg-black/50">
                    <img
                      src={previewUrl}
                      alt="Before"
                      className="w-full h-full object-cover"
                      onClick={() =>
                        openImageModal(previewUrl, {
                          stopPropagation: () => {},
                        })
                      }
                    />
                    <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full text-xs font-bold text-white border border-white/10">
                      BEFORE
                    </div>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="relative aspect-[3/4] rounded-xl overflow-hidden border-2 border-primary/50 bg-black/50 shadow-lg shadow-primary/10">
                    <img
                      src={resolveMediaUrl(overlayResult.overlay_url)}
                      alt="After"
                      className="w-full h-full object-cover cursor-pointer hover:scale-105 transition-transform duration-500"
                      onClick={() =>
                        openImageModal(
                          resolveMediaUrl(overlayResult.overlay_url),
                          { stopPropagation: () => {} }
                        )
                      }
                    />
                    <div className="absolute top-2 left-2 bg-primary/90 backdrop-blur-md px-3 py-1 rounded-full text-xs font-bold text-white shadow-lg">
                      AFTER
                    </div>
                  </div>
                </div>
              </div>
              <p className="text-center text-xs text-gray-500 mt-3">
                Click images to enlarge
              </p>
            </div>

            <div className="flex justify-center gap-4">
              <Button onClick={closeResultModal} variant="secondary">
                Close
              </Button>
              <Button onClick={handleSaveImage} variant="primary">
                Save Image
              </Button>
            </div>
          </div>
        )}
      </Modal>

      {/* Full Image Modal */}
      {showImageModal && (
        <div
          className="fixed inset-0 bg-black/95 z-[60] flex items-center justify-center p-4"
          onClick={closeImageModal}
        >
          <button
            className="absolute top-4 right-4 text-white/70 hover:text-white text-4xl font-light"
            onClick={closeImageModal}
          >
            &times;
          </button>
          <img
            src={modalImageUrl}
            alt="Full view"
            className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
};

export default Discover;
