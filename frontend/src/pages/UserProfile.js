import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import AuthService from '../services/AuthService';

const UserProfile = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [favorites, setFavorites] = useState([]);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    dateOfBirth: '',
    bio: ''
  });
  const [preferences, setPreferences] = useState({
    hairType: '',
    faceShape: '',
    lifestyle: '',
    maintenanceLevel: ''
  });
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        if (!currentUser) {
          navigate('/login');
          return;
        }
        setUser(currentUser);
        // Populate form with user data
        setFormData({
          firstName: currentUser.firstName || '',
          lastName: currentUser.lastName || '',
          email: currentUser.email || '',
          phone: currentUser.phone || '',
          dateOfBirth: currentUser.dateOfBirth || '',
          bio: currentUser.bio || ''
        });
        setPreferences({
          hairType: currentUser.hairType || '',
          faceShape: currentUser.faceShape || '',
          lifestyle: currentUser.lifestyle || '',
          maintenanceLevel: currentUser.maintenanceLevel || ''
        });
      } catch (error) {
        console.error('Authentication check failed:', error);
        navigate('/login');
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate]);

  const loadFavorites = () => {
    const storedFavorites = localStorage.getItem('hairstyle_favorites');
    if (storedFavorites) {
      setFavorites(JSON.parse(storedFavorites));
    }
  };

  const removeFavorite = (hairstyleId) => {
    const updatedFavorites = favorites.filter(fav => fav.id !== hairstyleId);
    setFavorites(updatedFavorites);
    localStorage.setItem('hairstyle_favorites', JSON.stringify(updatedFavorites));
  };

  useEffect(() => {
    loadFavorites();
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

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePreferenceChange = (e) => {
    const { name, value } = e.target;
    setPreferences(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSave = async () => {
    try {
      // Here you would typically make an API call to update user data
      console.log('Saving user data:', { ...formData, ...preferences });
      setIsEditing(false);
      // You can add a success message here
    } catch (error) {
      console.error('Failed to save user data:', error);
    }
  };

  const handleCancel = () => {
    // Reset form data to original user data
    setFormData({
      firstName: user?.firstName || '',
      lastName: user?.lastName || '',
      email: user?.email || '',
      phone: user?.phone || '',
      dateOfBirth: user?.dateOfBirth || '',
      bio: user?.bio || ''
    });
    setPreferences({
      hairType: user?.hairType || '',
      faceShape: user?.faceShape || '',
      lifestyle: user?.lifestyle || '',
      maintenanceLevel: user?.maintenanceLevel || ''
    });
    setIsEditing(false);
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
      <div className="min-h-screen bg-gray-900 pt-20 md:pt-24">
        {/* Header Section - Similar to Discover page */}
        <div className="bg-gradient-to-r from-purple-600 to-blue-600 py-12 md:py-16">
          <div className="max-w-7xl mx-auto px-4 text-center">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Your Profile
            </h1>
            <p className="text-xl text-gray-200 max-w-3xl mx-auto">
              Manage your account settings and preferences
            </p>
          </div>
        </div>

        <div className="max-w-4xl mx-auto px-4 py-8 md:py-12">
          {/* Profile Header */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 md:p-8 mb-8 border border-slate-700/50">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
              <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4 sm:gap-6">
                <div className="w-20 h-20 md:w-24 md:h-24 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white text-2xl md:text-3xl font-bold flex-shrink-0">
                  {user?.firstName?.charAt(0) || user?.email?.charAt(0) || 'U'}
                </div>
                <div className="text-center sm:text-left">
                  <h1 className="text-2xl md:text-3xl font-bold text-white mb-2">
                    {user?.firstName && user?.lastName 
                      ? `${user.firstName} ${user.lastName}` 
                      : user?.email || 'User Profile'
                    }
                  </h1>
                  <p className="text-gray-300 text-sm md:text-base">Member since {new Date().getFullYear()}</p>
                  <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2 md:gap-4 mt-2">
                    <span className="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-xs md:text-sm">
                      ✓ Verified
                    </span>
                    <span className="text-gray-400 text-xs md:text-sm">
                      {user?.analysisCount || 0} analyses completed
                    </span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setIsEditing(!isEditing)}
                className="w-full md:w-auto bg-purple-600 hover:bg-purple-700 text-white px-6 py-2.5 md:py-2 rounded-lg transition duration-300 font-medium flex-shrink-0"
              >
                {isEditing ? 'Cancel' : 'Edit Profile'}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">
            {/* Personal Information */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-slate-700/50">
              <h2 className="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6">Personal Information</h2>
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">First Name</label>
                    {isEditing ? (
                      <input
                        type="text"
                        name="firstName"
                        value={formData.firstName}
                        onChange={handleInputChange}
                        className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                      />
                    ) : (
                      <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg">
                        {formData.firstName || 'Not set'}
                      </p>
                    )}
                  </div>
                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Last Name</label>
                    {isEditing ? (
                      <input
                        type="text"
                        name="lastName"
                        value={formData.lastName}
                        onChange={handleInputChange}
                        className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                      />
                    ) : (
                      <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg">
                        {formData.lastName || 'Not set'}
                      </p>
                    )}
                  </div>
                </div>
                
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Email</label>
                  <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg">
                    {formData.email}
                  </p>
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Phone</label>
                  {isEditing ? (
                    <input
                      type="tel"
                      name="phone"
                      value={formData.phone}
                      onChange={handleInputChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    />
                  ) : (
                    <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg">
                      {formData.phone || 'Not set'}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Date of Birth</label>
                  {isEditing ? (
                    <input
                      type="date"
                      name="dateOfBirth"
                      value={formData.dateOfBirth}
                      onChange={handleInputChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    />
                  ) : (
                    <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg">
                      {formData.dateOfBirth || 'Not set'}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Bio</label>
                  {isEditing ? (
                    <textarea
                      name="bio"
                      value={formData.bio}
                      onChange={handleInputChange}
                      rows="3"
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none resize-none"
                      placeholder="Tell us about yourself..."
                    />
                  ) : (
                    <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg min-h-[80px]">
                      {formData.bio || 'No bio added yet'}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Hair & Style Preferences */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-slate-700/50">
              <h2 className="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6">Hair & Style Preferences</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Hair Type</label>
                  {isEditing ? (
                    <select
                      name="hairType"
                      value={preferences.hairType}
                      onChange={handlePreferenceChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="">Select hair type</option>
                      <option value="straight">Straight</option>
                      <option value="wavy">Wavy</option>
                      <option value="curly">Curly</option>
                      <option value="coily">Coily</option>
                    </select>
                  ) : (
                    <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg capitalize">
                      {preferences.hairType || 'Not set'}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Face Shape</label>
                  {isEditing ? (
                    <select
                      name="faceShape"
                      value={preferences.faceShape}
                      onChange={handlePreferenceChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="">Select face shape</option>
                      <option value="heart">Heart</option>
                      <option value="oblong">Oblong</option>
                      <option value="oval">Oval</option>
                      <option value="round">Round</option>
                      <option value="square">Square</option>
                    </select>
                  ) : (
                    <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg capitalize">
                      {preferences.faceShape || 'Not determined yet'}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Lifestyle</label>
                  {isEditing ? (
                    <select
                      name="lifestyle"
                      value={preferences.lifestyle}
                      onChange={handlePreferenceChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="">Select lifestyle</option>
                      <option value="active">Active/Sports</option>
                      <option value="professional">Professional</option>
                      <option value="casual">Casual</option>
                      <option value="creative">Creative</option>
                      <option value="social">Social</option>
                    </select>
                  ) : (
                    <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg capitalize">
                      {preferences.lifestyle || 'Not set'}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Maintenance Level</label>
                  {isEditing ? (
                    <select
                      name="maintenanceLevel"
                      value={preferences.maintenanceLevel}
                      onChange={handlePreferenceChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="">Select maintenance level</option>
                      <option value="low">Low - Minimal styling</option>
                      <option value="medium">Medium - Some styling required</option>
                      <option value="high">High - Daily styling needed</option>
                    </select>
                  ) : (
                    <p className="text-white bg-slate-700/50 px-4 py-2 rounded-lg capitalize">
                      {preferences.maintenanceLevel || 'Not set'}
                    </p>
                  )}
                </div>
              </div>

              {isEditing && (
                <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 mt-6">
                  <button
                    onClick={handleSave}
                    className="flex-1 bg-purple-600 hover:bg-purple-700 text-white py-2.5 md:py-2 px-4 rounded-lg transition duration-300 font-medium"
                  >
                    Save Changes
                  </button>
                  <button
                    onClick={handleCancel}
                    className="flex-1 bg-slate-600 hover:bg-slate-700 text-white py-2.5 md:py-2 px-4 rounded-lg transition duration-300 font-medium"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Favorite Hairstyles */}
          <div className="mt-6 md:mt-8 bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-slate-700/50">
            <h2 className="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6">Favorite Hairstyles</h2>
            
            {favorites.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {favorites.map((favorite) => (
                  <div 
                    key={favorite.id}
                    className="bg-slate-700/50 rounded-xl overflow-hidden border border-slate-600/50 hover:border-purple-500/50 transition-all duration-300"
                  >
                    <div className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <h3 className="text-white font-semibold text-lg flex-1 pr-2">
                          {favorite.name}
                        </h3>
                        <button
                          onClick={() => removeFavorite(favorite.id)}
                          className="text-red-500 hover:text-red-400 transition-colors duration-200 flex-shrink-0"
                          title="Remove from favorites"
                        >
                          <svg className="w-6 h-6 fill-current" viewBox="0 0 24 24">
                            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                          </svg>
                        </button>
                      </div>
                      
                      <div className="flex items-center text-gray-400 text-sm">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {new Date(favorite.timestamp).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric'
                        })}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <svg className="w-16 h-16 text-gray-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                </svg>
                <p className="text-gray-400 text-lg">No favorite hairstyles yet</p>
                <p className="text-gray-500 text-sm mt-2">Save hairstyles from the Results page by clicking the heart icon</p>
              </div>
            )}
          </div>

          {/* Recent Activity */}
          <div className="mt-6 md:mt-8 bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-slate-700/50">
            <h2 className="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6">Recent Activity</h2>
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 py-3 px-4 bg-slate-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white text-sm">📸</span>
                  </div>
                  <div>
                    <p className="text-white font-medium text-sm md:text-base">Photo analysis completed</p>
                    <p className="text-gray-400 text-xs md:text-sm">Received 6 hairstyle recommendations</p>
                  </div>
                </div>
                <span className="text-gray-400 text-xs md:text-sm sm:ml-auto">2 days ago</span>
              </div>
              
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 py-3 px-4 bg-slate-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white text-sm">⚙️</span>
                  </div>
                  <div>
                    <p className="text-white font-medium text-sm md:text-base">Preferences updated</p>
                    <p className="text-gray-400 text-xs md:text-sm">Updated lifestyle and maintenance preferences</p>
                  </div>
                </div>
                <span className="text-gray-400 text-xs md:text-sm sm:ml-auto">1 week ago</span>
              </div>
              
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 py-3 px-4 bg-slate-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white text-sm">✅</span>
                  </div>
                  <div>
                    <p className="text-white font-medium text-sm md:text-base">Account created</p>
                    <p className="text-gray-400 text-xs md:text-sm">Welcome to HairMixer!</p>
                  </div>
                </div>
                <span className="text-gray-400 text-xs md:text-sm sm:ml-auto">2 weeks ago</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default UserProfile;
