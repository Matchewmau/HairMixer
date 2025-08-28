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

  // Mock hairstyle data organized by categories
  const hairstyleCategories = {
    trending: {
      name: "Trending Now",
      icon: "🔥",
      styles: [
        {
          id: 1,
          name: "Modern Textured Bob",
          category: "trending",
          length: "Medium",
          maintenance: "Medium",
          theme: "Modern",
          description: "A contemporary take on the classic bob with added texture and movement.",
          features: ["Face-framing layers", "Textured finish", "Versatile styling"],
          suitableFor: ["Oval", "Heart", "Square"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 6-8 weeks",
          tags: ["Professional", "Casual", "Trendy"]
        },
        {
          id: 2,
          name: "Curtain Bangs Lob",
          category: "trending",
          length: "Medium",
          maintenance: "Low",
          theme: "Casual",
          description: "Long bob with soft curtain bangs for an effortless, chic look.",
          features: ["Curtain bangs", "Long bob cut", "Natural flow"],
          suitableFor: ["Oval", "Round", "Diamond"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 8-10 weeks",
          tags: ["Effortless", "Chic", "Low-maintenance"]
        }
      ]
    },
    classic: {
      name: "Classic Styles",
      icon: "👑",
      styles: [
        {
          id: 3,
          name: "Timeless Pixie Cut",
          category: "classic",
          length: "Short",
          maintenance: "High",
          theme: "Professional",
          description: "A sophisticated short cut that's both elegant and practical.",
          features: ["Clean lines", "Structured shape", "Professional look"],
          suitableFor: ["Oval", "Heart", "Diamond"],
          stylingTime: "5 minutes",
          maintenanceLevel: "Trim every 4-6 weeks",
          tags: ["Professional", "Elegant", "Low-styling"]
        },
        {
          id: 4,
          name: "Classic Long Layers",
          category: "classic",
          length: "Long",
          maintenance: "Medium",
          theme: "Versatile",
          description: "Timeless layered cut that works for any occasion.",
          features: ["Long layers", "Face-framing", "Volume boost"],
          suitableFor: ["All face shapes"],
          stylingTime: "15-20 minutes",
          maintenanceLevel: "Trim every 10-12 weeks",
          tags: ["Versatile", "Timeless", "Flattering"]
        }
      ]
    },
    edgy: {
      name: "Edgy & Bold",
      icon: "⚡",
      styles: [
        {
          id: 5,
          name: "Asymmetrical Bob",
          category: "edgy",
          length: "Short",
          maintenance: "High",
          theme: "Creative",
          description: "Bold asymmetrical cut for those who want to make a statement.",
          features: ["Asymmetrical length", "Sharp angles", "Modern edge"],
          suitableFor: ["Oval", "Square", "Heart"],
          stylingTime: "10-15 minutes",
          maintenanceLevel: "Trim every 4-6 weeks",
          tags: ["Bold", "Creative", "Statement"]
        },
        {
          id: 6,
          name: "Undercut Pixie",
          category: "edgy",
          length: "Short",
          maintenance: "High",
          theme: "Edgy",
          description: "Daring pixie cut with undercut details for maximum impact.",
          features: ["Undercut sides", "Textured top", "Bold contrast"],
          suitableFor: ["Oval", "Heart", "Diamond"],
          stylingTime: "5-10 minutes",
          maintenanceLevel: "Trim every 3-4 weeks",
          tags: ["Daring", "Bold", "High-impact"]
        }
      ]
    },
    lowMaintenance: {
      name: "Low Maintenance",
      icon: "🌿",
      styles: [
        {
          id: 7,
          name: "Natural Beach Waves",
          category: "lowMaintenance",
          length: "Medium",
          maintenance: "Low",
          theme: "Casual",
          description: "Effortless waves that look naturally beautiful with minimal styling.",
          features: ["Natural texture", "Minimal styling", "Air-dry friendly"],
          suitableFor: ["All face shapes"],
          stylingTime: "2-5 minutes",
          maintenanceLevel: "Trim every 12-16 weeks",
          tags: ["Natural", "Effortless", "Air-dry"]
        },
        {
          id: 8,
          name: "Wash & Go Curls",
          category: "lowMaintenance",
          length: "Medium",
          maintenance: "Low",
          theme: "Natural",
          description: "Embrace your natural curl pattern with this easy-care style.",
          features: ["Natural curls", "No heat styling", "Curl-enhancing"],
          suitableFor: ["Round", "Oval", "Heart"],
          stylingTime: "3-5 minutes",
          maintenanceLevel: "Trim every 10-14 weeks",
          tags: ["Natural", "Curl-friendly", "No-heat"]
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
      />
      <div className="min-h-screen bg-gray-900 pt-20">
        {/* Header Section */}
        <div className="bg-gradient-to-r from-purple-600 to-blue-600 py-16">
          <div className="max-w-7xl mx-auto px-4 text-center">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Discover Your Perfect Style
            </h1>
            <p className="text-xl text-gray-200 max-w-3xl mx-auto">
              Explore our curated collection of hairstyles across different categories. 
              Find inspiration for your next look!
            </p>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 py-12">
          {/* Category Filter */}
          <div className="mb-12">
            <h2 className="text-2xl font-bold text-white mb-6">Browse by Category</h2>
            <div className="flex flex-wrap gap-4">
              {categories.map((category) => (
                <button
                  key={category.key}
                  onClick={() => setSelectedCategory(category.key)}
                  className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition duration-300 ${
                    selectedCategory === category.key
                      ? 'bg-purple-600 text-white shadow-lg scale-105'
                      : 'bg-slate-800/50 text-gray-300 hover:bg-slate-700/50 hover:text-white'
                  }`}
                >
                  <span className="text-xl">{category.icon}</span>
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
                className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 hover:border-purple-500/30 transition-all duration-300 cursor-pointer group"
                onClick={() => openStyleDetails(style)}
              >
                {/* Style Image Placeholder */}
                <div className="w-full h-48 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-lg mb-4 flex items-center justify-center group-hover:scale-105 transition-transform duration-300">
                  <div className="text-4xl">💇‍♀️</div>
                </div>

                {/* Style Info */}
                <div className="space-y-3">
                  <h3 className="text-xl font-bold text-white group-hover:text-purple-400 transition-colors duration-300">
                    {style.name}
                  </h3>
                  <p className="text-gray-300 text-sm line-clamp-2">
                    {style.description}
                  </p>

                  {/* Style Attributes */}
                  <div className="flex flex-wrap gap-2">
                    <span className="bg-purple-900/30 text-purple-300 px-2 py-1 rounded-full text-xs">
                      {style.length}
                    </span>
                    <span className="bg-blue-900/30 text-blue-300 px-2 py-1 rounded-full text-xs">
                      {style.maintenance} Maintenance
                    </span>
                    <span className="bg-green-900/30 text-green-300 px-2 py-1 rounded-full text-xs">
                      {style.theme}
                    </span>
                  </div>

                  {/* View Details Button */}
                  <div className="pt-2">
                    <div className="text-purple-400 text-sm font-medium group-hover:text-purple-300 flex items-center">
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
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-slate-800 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              {/* Modal Header */}
              <div className="flex items-center justify-between p-6 border-b border-slate-700">
                <h2 className="text-2xl font-bold text-white">{selectedStyle.name}</h2>
                <button
                  onClick={closeStyleDetails}
                  className="text-gray-400 hover:text-white transition-colors duration-300"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Modal Content */}
              <div className="p-6 space-y-6">
                {/* Style Image */}
                <div className="w-full h-64 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-lg flex items-center justify-center">
                  <div className="text-6xl">💇‍♀️</div>
                </div>

                {/* Description */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">Description</h3>
                  <p className="text-gray-300">{selectedStyle.description}</p>
                </div>

                {/* Key Features */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Key Features</h3>
                  <ul className="space-y-2">
                    {selectedStyle.features.map((feature, index) => (
                      <li key={index} className="flex items-center text-gray-300">
                        <span className="text-purple-400 mr-2">•</span>
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Style Details Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-semibold text-purple-400 mb-1">Hair Length</h4>
                      <p className="text-white">{selectedStyle.length}</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-purple-400 mb-1">Styling Time</h4>
                      <p className="text-white">{selectedStyle.stylingTime}</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-purple-400 mb-1">Maintenance</h4>
                      <p className="text-white">{selectedStyle.maintenanceLevel}</p>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-semibold text-purple-400 mb-1">Best for Face Shapes</h4>
                      <p className="text-white">{selectedStyle.suitableFor.join(', ')}</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-purple-400 mb-1">Style Tags</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedStyle.tags.map((tag, index) => (
                          <span key={index} className="bg-purple-900/30 text-purple-300 px-2 py-1 rounded-full text-xs">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-4 pt-4">
                  <button
                    onClick={() => navigate('/upload')}
                    className="flex-1 bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-lg font-medium transition duration-300"
                  >
                    Try This Style
                  </button>
                  {user && (
                    <button className="flex-1 bg-slate-700 hover:bg-slate-600 text-white py-3 px-6 rounded-lg font-medium transition duration-300">
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
