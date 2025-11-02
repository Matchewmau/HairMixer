import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import AuthService from '../services/AuthService';

const Discover = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedStyle, setSelectedStyle] = useState(null);
  const navigate = useNavigate();

  // Static hairstyle data organized by categories with real descriptions and images
  const hairstyleCategories = {
    trending: {
      name: "Trending Now",
      icon: "🔥",
      styles: [
        {
          id: 1,
          name: "Butterfly Haircut",
          category: "trending",
          length: "Medium to Long",
          maintenance: "Medium",
          theme: "Modern",
          image: "/discover/butterfly-haircut.jpg",
          description: "The butterfly haircut features shorter layers around the crown that gradually blend into longer lengths, creating a beautiful winged effect. This viral TikTok trend adds volume and movement while maintaining length.",
          features: ["Layered crown", "Seamless blending", "Volume boost", "Face-framing effect"],
          suitableFor: ["Oval", "Round", "Heart", "Diamond"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 8-10 weeks",
          tags: ["TikTok Trend", "Voluminous", "Romantic"]
        },
        {
          id: 2,
          name: "Wolf Cut",
          category: "trending",
          length: "Medium",
          maintenance: "Low",
          theme: "Edgy",
          image: "/discover/wolfcut.jpg",
          description: "A hybrid of shag and mullet styles, the wolf cut features choppy layers throughout with shorter pieces on top and longer at the back. Perfect for those wanting an effortlessly cool, rock-inspired look.",
          features: ["Shaggy layers", "Textured finish", "Choppy bangs", "Mullet-inspired"],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 10-12 weeks",
          tags: ["Edgy", "Shaggy", "Low-maintenance"]
        },
        {
          id: 3,
          name: "Curtain Bangs with Long Layers",
          category: "trending",
          length: "Long",
          maintenance: "Low",
          theme: "Casual",
          image: "/discover/Curtain-Bangs-with-Long-Layers.jpg",
          description: "Soft, parted-down-the-middle bangs that frame the face beautifully, paired with long flowing layers. This 70s-inspired trend is flattering on everyone and easy to style.",
          features: ["Center-parted bangs", "Face-framing", "Soft layers", "Versatile styling"],
          suitableFor: ["All face shapes"],
          stylingTime: "8-12 minutes",
          maintenanceLevel: "Trim bangs every 4-6 weeks",
          tags: ["70s Inspired", "Face-framing", "Versatile"]
        },
        {
          id: 4,
          name: "Modern Shag",
          category: "trending",
          length: "Medium",
          maintenance: "Medium",
          theme: "Retro-Modern",
          image: "/discover/Modern-Shag.jpg",
          description: "An updated take on the classic shag with modern texturizing techniques. Features lots of layers, texture, and movement for an effortlessly cool vibe that works with any hair type.",
          features: ["Heavy layering", "Textured ends", "Wispy bangs option", "Volume throughout"],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 8-10 weeks",
          tags: ["Textured", "Retro", "Modern"]
        }
      ]
    },
    classic: {
      name: "Classic Styles",
      icon: "👑",
      styles: [
        {
          id: 5,
          name: "Timeless Pixie Cut",
          category: "classic",
          length: "Short",
          maintenance: "High",
          theme: "Sophisticated",
          image: "/discover/timeless-pixiecut.jpg",
          description: "A sophisticated short cut with tapered sides and back, and slightly longer top. Popularized by icons like Audrey Hepburn and Mia Farrow, this elegant style never goes out of fashion.",
          features: ["Clean lines", "Tapered sides", "Textured top", "Versatile styling"],
          suitableFor: ["Oval", "Heart", "Oblong"],
          stylingTime: "5-8 minutes",
          maintenanceLevel: "Trim every 4-6 weeks",
          tags: ["Iconic", "Elegant", "Professional"]
        },
        {
          id: 6,
          name: "Classic Bob",
          category: "classic",
          length: "Short to Medium",
          maintenance: "Medium",
          theme: "Timeless",
          image: "/discover/Classic-Bob-Cut.jpg",
          description: "The quintessential bob cut at chin or jaw length with clean, blunt ends. This versatile classic can be worn sleek and straight or with subtle waves for different occasions.",
          features: ["Blunt cut", "One-length", "Clean lines", "Sleek finish"],
          suitableFor: ["All face shapes"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 6-8 weeks",
          tags: ["Versatile", "Polished", "Timeless"]
        },
        {
          id: 7,
          name: "Long Layered Hair",
          category: "classic",
          length: "Long",
          maintenance: "Medium",
          theme: "Traditional",
          image: "https://images.unsplash.com/photo-1519699047748-de8e457a634e?w=500",
          description: "Long hair with subtle layers throughout to add movement and prevent heaviness. This timeless style flatters all face shapes and can be dressed up or down effortlessly.",
          features: ["Subtle layers", "Natural flow", "Face-framing", "Volume enhancement"],
          suitableFor: ["All face shapes"],
          stylingTime: "15-20 minutes",
          maintenanceLevel: "Trim every 10-12 weeks",
          tags: ["Traditional", "Feminine", "Elegant"]
        },
        {
          id: 8,
          name: "French Bob",
          category: "classic",
          length: "Short",
          maintenance: "High",
          theme: "Chic",
          image: "/discover/french-bob.jpg",
          description: "A chic, chin-length bob with a slightly tousled texture and often paired with bangs. This Parisian-inspired cut exudes effortless sophistication and timeless elegance.",
          features: ["Chin-length", "Slightly textured", "Optional bangs", "Effortless style"],
          suitableFor: ["Oval", "Heart", "Diamond"],
          stylingTime: "8-12 minutes",
          maintenanceLevel: "Trim every 6-8 weeks",
          tags: ["Parisian", "Chic", "Sophisticated"]
        }
      ]
    },
    edgy: {
      name: "Edgy & Bold",
      icon: "⚡",
      styles: [
        {
          id: 9,
          name: "Asymmetrical Bob",
          category: "edgy",
          length: "Short",
          maintenance: "High",
          theme: "Contemporary",
          image: "https://images.unsplash.com/photo-1522337660859-02fbefca4702?w=500",
          description: "A bold bob with dramatically different lengths on each side. One side typically sits at jaw-level while the other is cut shorter, creating a striking, fashion-forward statement.",
          features: ["Uneven lengths", "Sharp angles", "Modern edge", "Statement-making"],
          suitableFor: ["Oval", "Square", "Heart"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 4-6 weeks",
          tags: ["Bold", "Fashion-forward", "Dramatic"]
        },
        {
          id: 10,
          name: "Undercut with Long Top",
          category: "edgy",
          length: "Short to Medium",
          maintenance: "High",
          theme: "Edgy",
          image: "/discover/undercut-longtop.jpg",
          description: "Shaved or closely cropped sides and back with longer hair on top that can be styled in various ways. This contrasting style offers versatility and a bold, modern aesthetic.",
          features: ["Shaved sides", "Long top section", "High contrast", "Versatile styling"],
          suitableFor: ["Oval", "Heart", "Oblong", "Square"],
          stylingTime: "5-15 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Bold", "Versatile", "Modern"]
        },
        {
          id: 11,
          name: "Platinum Buzz Cut",
          category: "edgy",
          length: "Very Short",
          maintenance: "High",
          theme: "Bold",
          image: "https://images.unsplash.com/photo-1526510747491-58f928ec870f?w=500",
          description: "An ultra-short buzz cut often paired with platinum blonde or bold color. This fearless style is low-maintenance for styling but requires regular touch-ups to maintain the color and length.",
          features: ["Ultra-short length", "Bold color option", "Clean aesthetic", "Confidence-boosting"],
          suitableFor: ["Oval", "Heart", "Diamond"],
          stylingTime: "2-3 minutes",
          maintenanceLevel: "Trim every 2-3 weeks, color every 4-6 weeks",
          tags: ["Fearless", "Low-styling", "Statement"]
        },
        {
          id: 12,
          name: "Mohawk Fade",
          category: "edgy",
          length: "Short",
          maintenance: "High",
          theme: "Punk-Inspired",
          image: "/discover/fade-mohawk.jpg",
          description: "A modern take on the mohawk with faded sides and a styled strip of hair down the center. Can be worn sleek or textured, offering a punk-rock edge with contemporary polish.",
          features: ["Center strip", "Faded sides", "Textured top", "Statement style"],
          suitableFor: ["Oval", "Oblong", "Heart"],
          stylingTime: "10-20 minutes",
          maintenanceLevel: "Trim every 2-4 weeks",
          tags: ["Punk", "Bold", "Unique"]
        }
      ]
    },
    lowMaintenance: {
      name: "Low Maintenance",
      icon: "🌿",
      styles: [
        {
          id: 13,
          name: "Natural Beach Waves",
          category: "lowMaintenance",
          length: "Medium to Long",
          maintenance: "Low",
          theme: "Casual",
          image: "https://images.unsplash.com/photo-1573007974656-b958089e9f7b?w=500",
          description: "Loose, natural-looking waves that require minimal heat styling. Achieved through braiding, twisting, or sea salt spray for that effortless, sun-kissed beach look year-round.",
          features: ["Natural texture", "Air-dry friendly", "Sea salt spray", "Effortless vibe"],
          suitableFor: ["All face shapes"],
          stylingTime: "2-5 minutes",
          maintenanceLevel: "Trim every 12-16 weeks",
          tags: ["Natural", "Beachy", "Effortless"]
        },
        {
          id: 14,
          name: "Wash and Go Curls",
          category: "lowMaintenance",
          length: "Any",
          maintenance: "Low",
          theme: "Natural",
          image: "/discover/wash-n-go-curls.jpg",
          description: "Embrace your natural curl pattern with curl-enhancing products and minimal manipulation. This healthy approach celebrates natural texture while maintaining gorgeous, defined curls.",
          features: ["Natural curls", "Curl-defining products", "No heat styling", "Healthy hair focus"],
          suitableFor: ["Round", "Oval", "Heart", "Diamond"],
          stylingTime: "3-8 minutes",
          maintenanceLevel: "Trim every 10-14 weeks",
          tags: ["Natural", "Curly", "Healthy"]
        },
        {
          id: 15,
          name: "Blunt Long Hair",
          category: "lowMaintenance",
          length: "Long",
          maintenance: "Low",
          theme: "Minimalist",
          image: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=500",
          description: "Simple, one-length long hair with blunt ends. This minimalist style requires little daily maintenance and can be worn straight, wavy, or in various updos with ease.",
          features: ["One-length", "Blunt ends", "Versatile styling", "Minimal maintenance"],
          suitableFor: ["All face shapes"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 12-16 weeks",
          tags: ["Simple", "Versatile", "Classic"]
        },
        {
          id: 16,
          name: "Shoulder-Length Straight",
          category: "lowMaintenance",
          length: "Medium",
          maintenance: "Low",
          theme: "Practical",
          image: "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=500",
          description: "Straight hair cut to shoulder length with minimal layers. This practical style air-dries well, requires minimal styling, and is perfect for busy lifestyles while still looking polished.",
          features: ["Shoulder-length", "Straight cut", "Air-dry friendly", "Low styling"],
          suitableFor: ["All face shapes"],
          stylingTime: "5-8 minutes",
          maintenanceLevel: "Trim every 10-12 weeks",
          tags: ["Practical", "Easy", "Polished"]
        }
      ]
    }
  };

  const categories = [
    { key: 'all', name: 'All Styles', icon: '🎨' },
    { key: 'trending', name: 'Trending', icon: '🔥' },
    { key: 'classic', name: 'Classic', icon: '👑' },
    { key: 'edgy', name: 'Edgy', icon: '⚡' },
    { key: 'lowMaintenance', name: 'Low Maintenance', icon: '🌿' }
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
                  <div className="w-full h-56 bg-gradient-to-br from-purple-600/20 to-blue-600/20 flex items-center justify-center group-hover:scale-110 transition-transform duration-500">
                    {style.image ? (
                      <img 
                        src={style.image} 
                        alt={style.name}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.target.style.display = 'none';
                          e.target.nextElementSibling.style.display = 'flex';
                        }}
                      />
                    ) : (
                      <div className="text-5xl">💇‍♀️</div>
                    )}
                    <div className="hidden text-5xl items-center justify-center w-full h-full bg-gradient-to-br from-purple-600/20 to-blue-600/20">💇‍♀️</div>
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
          <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-white/10 shadow-2xl">
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
                <div className="w-full h-72 bg-gradient-to-br from-purple-600/20 to-blue-600/20 rounded-xl flex items-center justify-center border border-white/10 overflow-hidden">
                  {selectedStyle.image ? (
                    <img 
                      src={selectedStyle.image} 
                      alt={selectedStyle.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextElementSibling.style.display = 'flex';
                      }}
                    />
                  ) : (
                    <div className="text-7xl">💇‍♀️</div>
                  )}
                  <div className="hidden text-7xl items-center justify-center w-full h-full bg-gradient-to-br from-purple-600/20 to-blue-600/20">💇‍♀️</div>
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
                    onClick={() => navigate('/upload')}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-4 px-6 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-purple-500/25 border border-blue-500/30"
                  >
                    Try This Style
                  </button>
                  {user && (
                    <button className="flex-1 bg-white/10 hover:bg-white/20 text-white py-4 px-6 rounded-lg font-semibold transition-all duration-300 backdrop-blur-sm border border-white/20 hover:border-white/40">
                      Save to Favorites
                    </button>
                  )}
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
