import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import AuthService from '../services/AuthService';
import apiService from '../services/api';

const UserProfile = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [savedHairstyles, setSavedHairstyles] = useState([]);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: ''
  });
  
  // Preference Profiles state
  const [preferenceProfiles, setPreferenceProfiles] = useState([]);
  const [showCreateProfileModal, setShowCreateProfileModal] = useState(false);
  const [showEditProfileModal, setShowEditProfileModal] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [profileForm, setProfileForm] = useState({
    profile_name: '',
    description: '',
    gender: 'female',
    hair_type: 'straight',
    hair_length: 'medium',
    volume: 'medium',
    hair_thickness: 'medium',
    hair_texture_detail: 'normal',
    lifestyle: 'casual',
    maintenance: 'medium',
    styling_preference: 'natural',
    hair_color: 'brown',
    hair_condition: [],
    occasion: 'casual',
  });
  
  const navigate = useNavigate();
  
  // State for viewing saved hairstyle details
  const [showSavedModal, setShowSavedModal] = useState(false);
  const [selectedSaved, setSelectedSaved] = useState(null);

  // Load preference profiles
  const loadPreferenceProfiles = useCallback(async () => {
    try {
      const response = await apiService.getPreferenceProfiles();
      setPreferenceProfiles(response.profiles || []);
    } catch (error) {
      console.error('Failed to load preference profiles:', error);
    }
  }, []);

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
          email: currentUser.email || ''
        });
        
        // Load preference profiles
        await loadPreferenceProfiles();
      } catch (error) {
        console.error('Authentication check failed:', error);
        navigate('/login');
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate, loadPreferenceProfiles]);

  const loadSavedHairstyles = useCallback(() => {
    if (!user?.id) return;
    
    const storageKey = `saved_hairstyle_recommendations_user_${user.id}`;
    const storedSaved = localStorage.getItem(storageKey);
    if (storedSaved) {
      setSavedHairstyles(JSON.parse(storedSaved));
    }
  }, [user?.id]);

  const removeSavedHairstyle = (hairstyleId) => {
    if (!user?.id) return;
    
    const updatedSaved = savedHairstyles.filter(saved => saved.id !== hairstyleId);
    setSavedHairstyles(updatedSaved);
    const storageKey = `saved_hairstyle_recommendations_user_${user.id}`;
    localStorage.setItem(storageKey, JSON.stringify(updatedSaved));
  };

  const handleViewSaved = (saved) => {
    setSelectedSaved(saved);
    setShowSavedModal(true);
  };

  const closeSavedModal = () => {
    setShowSavedModal(false);
    setSelectedSaved(null);
  };

  useEffect(() => {
    if (user) {
      loadSavedHairstyles();
    }
  }, [user, loadSavedHairstyles]);

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

  const handleProfileFormChange = (e) => {
    const { name, value } = e.target;
    setProfileForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSave = async () => {
    try {
      await apiService.updateUserProfile(formData);
      setUser({ ...user, ...formData });
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to save user data:', error);
      alert('Failed to save profile. Please try again.');
    }
  };

  const handleCancel = () => {
    // Reset form data to original user data
    setFormData({
      firstName: user?.firstName || '',
      lastName: user?.lastName || '',
      email: user?.email || ''
    });
    setIsEditing(false);
  };

  // Preference Profile handlers
  const handleCreateProfile = async () => {
    try {
      await apiService.createPreferenceProfile(profileForm);
      await loadPreferenceProfiles();
      setShowCreateProfileModal(false);
      // Reset form
      setProfileForm({
        profile_name: '',
        description: '',
        gender: 'female',
        hair_type: 'straight',
        hair_length: 'medium',
        volume: 'medium',
        hair_thickness: 'medium',
        hair_texture_detail: 'normal',
        lifestyle: 'casual',
        maintenance: 'medium',
        styling_preference: 'natural',
        hair_color: 'brown',
        hair_condition: [],
        occasion: 'casual',
      });
    } catch (error) {
      console.error('Failed to create profile:', error);
      alert(error.message || 'Failed to create profile. Please try again.');
    }
  };

  const handleEditProfile = async () => {
    try {
      await apiService.updatePreferenceProfile(selectedProfile.id, profileForm);
      await loadPreferenceProfiles();
      setShowEditProfileModal(false);
      setSelectedProfile(null);
    } catch (error) {
      console.error('Failed to update profile:', error);
      alert(error.message || 'Failed to update profile. Please try again.');
    }
  };

  const handleDeleteProfile = async (profileId) => {
    if (!window.confirm('Are you sure you want to delete this profile?')) return;
    try {
      await apiService.deletePreferenceProfile(profileId);
      await loadPreferenceProfiles();
    } catch (error) {
      console.error('Failed to delete profile:', error);
      alert('Failed to delete profile. Please try again.');
    }
  };

  const handleSetDefault = async (profileId) => {
    try {
      await apiService.setDefaultProfile(profileId);
      await loadPreferenceProfiles();
    } catch (error) {
      console.error('Failed to set default profile:', error);
      alert('Failed to set default profile. Please try again.');
    }
  };

  const openEditModal = (profile) => {
    setSelectedProfile(profile);
    setProfileForm({
      profile_name: profile.profile_name,
      description: profile.description || '',
      gender: profile.gender,
      hair_type: profile.hair_type,
      hair_length: profile.hair_length,
      volume: profile.volume || 'medium',
      hair_thickness: profile.hair_thickness || 'medium',
      hair_texture_detail: profile.hair_texture_detail || 'normal',
      lifestyle: profile.lifestyle,
      maintenance: profile.maintenance || 'medium',
      styling_preference: profile.styling_preference || 'natural',
      hair_color: profile.hair_color || 'brown',
      hair_condition: profile.hair_condition || [],
      occasion: profile.occasion || 'casual',
    });
    setShowEditProfileModal(true);
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
          {/* Profile Header with Personal Info */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 md:p-8 mb-8 border border-slate-700/50">
            <div className="flex flex-col md:flex-row md:items-start gap-6">
              {/* Avatar */}
              <div className="flex justify-center md:justify-start">
                <div className="w-24 h-24 md:w-32 md:h-32 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white text-3xl md:text-4xl font-bold flex-shrink-0">
                  {user?.firstName?.charAt(0) || user?.email?.charAt(0) || 'U'}
                </div>
              </div>

              {/* Profile Information */}
              <div className="flex-1 text-center md:text-left">
                {isEditing ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-gray-400 text-xs font-medium mb-1.5">First Name</label>
                        <input
                          type="text"
                          name="firstName"
                          value={formData.firstName}
                          onChange={handleInputChange}
                          placeholder="Enter first name"
                          className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="block text-gray-400 text-xs font-medium mb-1.5">Last Name</label>
                        <input
                          type="text"
                          name="lastName"
                          value={formData.lastName}
                          onChange={handleInputChange}
                          placeholder="Enter last name"
                          className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                        />
                      </div>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-3 pt-2">
                      <button
                        onClick={handleSave}
                        className="flex-1 bg-purple-600 hover:bg-purple-700 text-white py-2.5 px-6 rounded-lg transition duration-300 font-medium"
                      >
                        Save Changes
                      </button>
                      <button
                        onClick={handleCancel}
                        className="flex-1 bg-slate-600 hover:bg-slate-700 text-white py-2.5 px-6 rounded-lg transition duration-300 font-medium"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mb-3">
                      <h1 className="text-2xl md:text-3xl font-bold text-white mb-1">
                        {user?.firstName && user?.lastName 
                          ? `${user.firstName} ${user.lastName}` 
                          : user?.email || 'User Profile'
                        }
                      </h1>
                    </div>
                    
                    <div className="space-y-2 mb-4">
                      <div className="flex items-center justify-center md:justify-start text-gray-300">
                        <svg className="w-5 h-5 mr-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                        <span className="text-sm md:text-base">{user?.email}</span>
                      </div>
                      
                      <div className="flex items-center justify-center md:justify-start text-gray-300">
                        <svg className="w-5 h-5 mr-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <span className="text-sm md:text-base">
                          Joined {user?.dateJoined ? new Date(user.dateJoined).toLocaleDateString('en-US', {
                            month: 'long',
                            year: 'numeric'
                          }) : new Date().toLocaleDateString('en-US', {
                            month: 'long',
                            year: 'numeric'
                          })}
                        </span>
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Edit Button */}
              {!isEditing && (
                <div className="flex justify-center md:justify-end">
                  <button
                    onClick={() => setIsEditing(true)}
                    className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded-lg transition duration-300 font-medium flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                    <span>Edit Profile</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 md:gap-8">

            {/* Preference Profiles */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-slate-700/50">
              <div className="flex items-center justify-between mb-4 md:mb-6">
                <div>
                  <h2 className="text-xl md:text-2xl font-bold text-white">Preference Profiles</h2>
                  <p className="text-gray-400 text-sm mt-1">Create and manage your hairstyle preference profiles</p>
                </div>
                <button
                  onClick={() => setShowCreateProfileModal(true)}
                  className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition duration-300 font-medium flex items-center gap-2"
                >
                  <span className="text-xl">+</span>
                  <span className="hidden sm:inline">Create Profile</span>
                </button>
              </div>

              {preferenceProfiles.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {preferenceProfiles.map((profile) => (
                    <div 
                      key={profile.id}
                      className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 hover:border-purple-500/50 transition-all duration-200"
                    >
                      {/* Header */}
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <h3 className="text-base font-semibold text-white mb-1">
                            {profile.profile_name}
                          </h3>
                          {profile.description && (
                            <p className="text-gray-400 text-xs line-clamp-2">{profile.description}</p>
                          )}
                        </div>
                        {profile.is_default && (
                          <span className="px-2 py-0.5 bg-purple-500/20 text-purple-300 text-xs font-medium rounded border border-purple-500/30">
                            Default
                          </span>
                        )}
                      </div>
                      
                      {/* Preference Info */}
                      <div className="space-y-2 mb-3 text-xs">
                        <div className="flex items-center justify-between text-gray-300">
                          <span className="text-gray-500">Gender:</span>
                          <span className="capitalize">{profile.gender}</span>
                        </div>
                        <div className="flex items-center justify-between text-gray-300">
                          <span className="text-gray-500">Hair Type:</span>
                          <span className="capitalize">{profile.hair_type}</span>
                        </div>
                        <div className="flex items-center justify-between text-gray-300">
                          <span className="text-gray-500">Length:</span>
                          <span className="capitalize">{profile.hair_length}</span>
                        </div>
                        <div className="flex items-center justify-between text-gray-300">
                          <span className="text-gray-500">Lifestyle:</span>
                          <span className="capitalize">{profile.lifestyle}</span>
                        </div>
                      </div>
                      
                      {/* Action Buttons */}
                      <div className="flex gap-2 pt-3 border-t border-slate-700">
                        <button
                          onClick={() => openEditModal(profile)}
                          className="flex-1 bg-slate-700 hover:bg-slate-600 text-white px-3 py-1.5 rounded text-xs font-medium transition-colors"
                        >
                          Edit
                        </button>
                        {!profile.is_default && (
                          <button
                            onClick={() => handleSetDefault(profile.id)}
                            className="flex-1 bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 px-3 py-1.5 rounded text-xs font-medium transition-colors border border-purple-500/30"
                          >
                            Set Default
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteProfile(profile.id)}
                          className="bg-red-600/20 hover:bg-red-600/30 text-red-400 px-3 py-1.5 rounded text-xs font-medium transition-colors border border-red-500/30"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <p className="text-gray-400 text-base mb-4">No preference profiles yet</p>
                  <button
                    onClick={() => setShowCreateProfileModal(true)}
                    className="inline-flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-5 py-2.5 rounded-lg transition duration-200 font-medium"
                  >
                    Create Profile
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Saved Hairstyle Recommendations */}
          <div className="mt-6 md:mt-8 bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-slate-700/50">
            <h2 className="text-xl md:text-2xl font-bold text-white mb-2">Saved Hairstyle Recommendations</h2>
            <p className="text-gray-400 text-sm mb-4 md:mb-6">Your personalized hairstyle recommendations given by the system</p>
            
            {savedHairstyles.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {savedHairstyles.map((saved) => (
                  <div 
                    key={saved.id}
                    className="bg-slate-700/50 rounded-xl overflow-hidden border border-slate-600/50 hover:border-blue-500/50 transition-all duration-300 cursor-pointer group"
                    onClick={() => handleViewSaved(saved)}
                  >
                    {/* Show overlay image if available */}
                    {saved.overlay_url && (
                      <div className="relative h-48 overflow-hidden bg-gradient-to-br from-purple-600/20 to-blue-600/20">
                        <img 
                          src={saved.overlay_url}
                          alt={saved.name}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        />
                      </div>
                    )}
                    
                    <div className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <h3 className="text-white font-semibold text-lg flex-1 pr-2 group-hover:text-blue-400 transition-colors">
                          {saved.name}
                        </h3>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            removeSavedHairstyle(saved.id);
                          }}
                          className="text-red-500 hover:text-red-400 transition-colors duration-200 flex-shrink-0"
                          title="Remove saved hairstyle"
                        >
                          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                      
                      <div className="flex items-center text-gray-400 text-sm mb-2">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {new Date(saved.saved_at).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric'
                        })}
                      </div>
                      
                      <p className="text-purple-400 text-sm group-hover:text-purple-300 transition-colors">
                        Click to view details →
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <svg className="w-16 h-16 text-gray-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z" />
                </svg>
                <p className="text-gray-400 text-lg">No saved hairstyle recommendations yet</p>
                <p className="text-gray-500 text-sm mt-2">Save hairstyle recommendations from the Results page by clicking the Save button</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Create Preference Profile Modal */}
      {showCreateProfileModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl max-w-2xl w-full my-8">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Create Preference Profile</h2>
                <button
                  onClick={() => setShowCreateProfileModal(false)}
                  className="text-gray-400 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Profile Name *</label>
                  <input
                    type="text"
                    name="profile_name"
                    value={profileForm.profile_name}
                    onChange={handleProfileFormChange}
                    placeholder="e.g., Professional Look, Casual Style"
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Description (Optional)</label>
                  <textarea
                    name="description"
                    value={profileForm.description}
                    onChange={handleProfileFormChange}
                    placeholder="Describe when you'd use this profile..."
                    rows="2"
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Gender *</label>
                    <select
                      name="gender"
                      value={profileForm.gender}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Type *</label>
                    <select
                      name="hair_type"
                      value={profileForm.hair_type}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="straight">Straight</option>
                      <option value="wavy">Wavy</option>
                      <option value="curly">Curly</option>
                      <option value="coily">Coily</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Length *</label>
                    <select
                      name="hair_length"
                      value={profileForm.hair_length}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="short">Short</option>
                      <option value="medium">Medium</option>
                      <option value="long">Long</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Volume *</label>
                    <select
                      name="volume"
                      value={profileForm.volume}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Thickness *</label>
                    <select
                      name="hair_thickness"
                      value={profileForm.hair_thickness}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="thin">Thin</option>
                      <option value="medium">Medium</option>
                      <option value="thick">Thick</option>
                      <option value="very_thick">Very Thick</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Texture *</label>
                    <select
                      name="hair_texture_detail"
                      value={profileForm.hair_texture_detail}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="fine">Fine</option>
                      <option value="normal">Normal</option>
                      <option value="thick">Thick</option>
                      <option value="smooth">Smooth</option>
                      <option value="coarse">Coarse</option>
                      <option value="silky">Silky</option>
                      <option value="frizzy">Frizzy</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Lifestyle *</label>
                    <select
                      name="lifestyle"
                      value={profileForm.lifestyle}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="active">Active</option>
                      <option value="professional">Professional</option>
                      <option value="creative">Creative</option>
                      <option value="casual">Casual</option>
                      <option value="moderate">Moderate</option>
                      <option value="relaxed">Relaxed</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Maintenance *</label>
                    <select
                      name="maintenance"
                      value={profileForm.maintenance}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Styling Preference *</label>
                    <select
                      name="styling_preference"
                      value={profileForm.styling_preference}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="natural">Natural</option>
                      <option value="casual">Casual</option>
                      <option value="classic">Classic</option>
                      <option value="polished">Polished</option>
                      <option value="elegant">Elegant</option>
                      <option value="glamorous">Glamorous</option>
                      <option value="trendy">Trendy</option>
                      <option value="edgy">Edgy</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Color *</label>
                    <select
                      name="hair_color"
                      value={profileForm.hair_color}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="black">Black</option>
                      <option value="brown">Brown</option>
                      <option value="blonde">Blonde</option>
                      <option value="red">Red</option>
                      <option value="auburn">Auburn</option>
                      <option value="gray">Gray</option>
                      <option value="white">White</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Occasion *</label>
                    <select
                      name="occasion"
                      value={profileForm.occasion}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="casual">Casual</option>
                      <option value="formal">Formal</option>
                      <option value="professional">Professional</option>
                      <option value="party">Party</option>
                      <option value="wedding">Wedding</option>
                      <option value="date">Date</option>
                      <option value="everyday">Everyday</option>
                      <option value="special_event">Special Event</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Hair Condition (Optional - Select multiple if needed)</label>
                  <div className="text-xs text-gray-400 mb-2">Select any conditions that apply to your hair</div>
                  <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto p-2 bg-slate-800 rounded-lg">
                    {['none', 'excellent', 'good', 'fair', 'damaged', 'dry_ends', 'oily_scalp', 'dandruff', 'frizzy', 'split_ends', 'thinning', 'sensitive_scalp'].map((condition) => (
                      <label key={condition} className="flex items-center space-x-2 text-sm text-gray-300 hover:text-white cursor-pointer">
                        <input
                          type="checkbox"
                          checked={Array.isArray(profileForm.hair_condition) && profileForm.hair_condition.includes(condition)}
                          onChange={(e) => {
                            const newConditions = e.target.checked
                              ? [...(Array.isArray(profileForm.hair_condition) ? profileForm.hair_condition : []), condition]
                              : (Array.isArray(profileForm.hair_condition) ? profileForm.hair_condition : []).filter(c => c !== condition);
                            handleProfileFormChange({ target: { name: 'hair_condition', value: newConditions } });
                          }}
                          className="form-checkbox h-4 w-4 text-purple-600 rounded"
                        />
                        <span className="capitalize">{condition.replace('_', ' ')}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={handleCreateProfile}
                  disabled={!profileForm.profile_name}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-2 rounded-lg transition font-medium"
                >
                  Create Profile
                </button>
                <button
                  onClick={() => setShowCreateProfileModal(false)}
                  className="flex-1 bg-slate-600 hover:bg-slate-700 text-white py-2 rounded-lg transition font-medium"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Edit Preference Profile Modal */}
      {showEditProfileModal && selectedProfile && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl max-w-2xl w-full my-8">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Edit Preference Profile</h2>
                <button
                  onClick={() => {
                    setShowEditProfileModal(false);
                    setSelectedProfile(null);
                  }}
                  className="text-gray-400 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Profile Name *</label>
                  <input
                    type="text"
                    name="profile_name"
                    value={profileForm.profile_name}
                    onChange={handleProfileFormChange}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Description (Optional)</label>
                  <textarea
                    name="description"
                    value={profileForm.description}
                    onChange={handleProfileFormChange}
                    rows="2"
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Gender *</label>
                    <select
                      name="gender"
                      value={profileForm.gender}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Type *</label>
                    <select
                      name="hair_type"
                      value={profileForm.hair_type}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="straight">Straight</option>
                      <option value="wavy">Wavy</option>
                      <option value="curly">Curly</option>
                      <option value="coily">Coily</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Length *</label>
                    <select
                      name="hair_length"
                      value={profileForm.hair_length}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="short">Short</option>
                      <option value="medium">Medium</option>
                      <option value="long">Long</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Volume *</label>
                    <select
                      name="volume"
                      value={profileForm.volume}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Thickness *</label>
                    <select
                      name="hair_thickness"
                      value={profileForm.hair_thickness}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="thin">Thin</option>
                      <option value="medium">Medium</option>
                      <option value="thick">Thick</option>
                      <option value="very_thick">Very Thick</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Texture *</label>
                    <select
                      name="hair_texture_detail"
                      value={profileForm.hair_texture_detail}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="fine">Fine</option>
                      <option value="normal">Normal</option>
                      <option value="thick">Thick</option>
                      <option value="smooth">Smooth</option>
                      <option value="coarse">Coarse</option>
                      <option value="silky">Silky</option>
                      <option value="frizzy">Frizzy</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Lifestyle *</label>
                    <select
                      name="lifestyle"
                      value={profileForm.lifestyle}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="active">Active</option>
                      <option value="professional">Professional</option>
                      <option value="creative">Creative</option>
                      <option value="casual">Casual</option>
                      <option value="moderate">Moderate</option>
                      <option value="relaxed">Relaxed</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Maintenance *</label>
                    <select
                      name="maintenance"
                      value={profileForm.maintenance}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Styling Preference *</label>
                    <select
                      name="styling_preference"
                      value={profileForm.styling_preference}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="natural">Natural</option>
                      <option value="casual">Casual</option>
                      <option value="classic">Classic</option>
                      <option value="polished">Polished</option>
                      <option value="elegant">Elegant</option>
                      <option value="glamorous">Glamorous</option>
                      <option value="trendy">Trendy</option>
                      <option value="edgy">Edgy</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Hair Color *</label>
                    <select
                      name="hair_color"
                      value={profileForm.hair_color}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="black">Black</option>
                      <option value="brown">Brown</option>
                      <option value="blonde">Blonde</option>
                      <option value="red">Red</option>
                      <option value="auburn">Auburn</option>
                      <option value="gray">Gray</option>
                      <option value="white">White</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm font-medium mb-2">Occasion *</label>
                    <select
                      name="occasion"
                      value={profileForm.occasion}
                      onChange={handleProfileFormChange}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="casual">Casual</option>
                      <option value="formal">Formal</option>
                      <option value="professional">Professional</option>
                      <option value="party">Party</option>
                      <option value="wedding">Wedding</option>
                      <option value="date">Date</option>
                      <option value="everyday">Everyday</option>
                      <option value="special_event">Special Event</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">Hair Condition (Optional - Select multiple if needed)</label>
                  <div className="text-xs text-gray-400 mb-2">Select any conditions that apply to your hair</div>
                  <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto p-2 bg-slate-800 rounded-lg">
                    {['none', 'excellent', 'good', 'fair', 'damaged', 'dry_ends', 'oily_scalp', 'dandruff', 'frizzy', 'split_ends', 'thinning', 'sensitive_scalp'].map((condition) => (
                      <label key={condition} className="flex items-center space-x-2 text-sm text-gray-300 hover:text-white cursor-pointer">
                        <input
                          type="checkbox"
                          checked={Array.isArray(profileForm.hair_condition) && profileForm.hair_condition.includes(condition)}
                          onChange={(e) => {
                            const newConditions = e.target.checked
                              ? [...(Array.isArray(profileForm.hair_condition) ? profileForm.hair_condition : []), condition]
                              : (Array.isArray(profileForm.hair_condition) ? profileForm.hair_condition : []).filter(c => c !== condition);
                            handleProfileFormChange({ target: { name: 'hair_condition', value: newConditions } });
                          }}
                          className="form-checkbox h-4 w-4 text-purple-600 rounded"
                        />
                        <span className="capitalize">{condition.replace('_', ' ')}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={handleEditProfile}
                  disabled={!profileForm.profile_name}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-2 rounded-lg transition font-medium"
                >
                  Save Changes
                </button>
                <button
                  onClick={() => {
                    setShowEditProfileModal(false);
                    setSelectedProfile(null);
                  }}
                  className="flex-1 bg-slate-600 hover:bg-slate-700 text-white py-2 rounded-lg transition font-medium"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Saved Hairstyle Recommendation Details Modal */}
      {showSavedModal && selectedSaved && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl max-w-4xl w-full my-8">
            <div className="p-6">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">
                  {selectedSaved.name}
                </h2>
                <button
                  onClick={closeSavedModal}
                  className="text-gray-400 hover:text-white text-2xl"
                  aria-label="Close"
                >
                  ✕
                </button>
              </div>

              {/* Content */}
              <div className="space-y-4 overflow-y-auto max-h-[600px]">
                {/* AI Generated Overlay Image */}
                {selectedSaved.overlay_url && (
                  <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-500/30 rounded-xl p-4">
                    <h3 className="text-lg font-semibold text-white mb-3">AI-Generated Preview</h3>
                    <div className="flex justify-center">
                      <img
                        src={selectedSaved.overlay_url}
                        alt={`${selectedSaved.name} overlay preview`}
                        className="max-w-md w-full rounded-lg shadow-lg"
                      />
                    </div>
                  </div>
                )}

                {/* Personalized Description */}
                {selectedSaved.personalized_description && (
                  <div className="bg-blue-900/20 border border-blue-500/30 rounded-xl p-4">
                    <h3 className="text-lg font-semibold text-white mb-3">✨ Why This Style Works For You</h3>
                    <p className="text-gray-300 leading-relaxed whitespace-pre-line">
                      {selectedSaved.personalized_description}
                    </p>
                  </div>
                )}

                {/* Face Shape Info */}
                {selectedSaved.face_shape && (
                  <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-4">
                    <h3 className="text-lg font-semibold text-white mb-2">Face Shape Analysis</h3>
                    <p className="text-green-300 capitalize">
                      Your face shape: <span className="font-semibold">{selectedSaved.face_shape}</span>
                    </p>
                    <p className="text-gray-400 text-sm mt-1">
                      This hairstyle was selected based on your unique facial features
                    </p>
                  </div>
                )}

                {/* User Preferences Used */}
                {selectedSaved.user_preferences && (
                  <div className="bg-purple-900/20 border border-purple-500/30 rounded-xl p-4">
                    <h3 className="text-lg font-semibold text-white mb-3">👤 Your Preferences</h3>
                    <div className="grid grid-cols-2 gap-3">
                      {selectedSaved.user_preferences.gender && (
                        <div className="text-sm">
                          <span className="text-gray-400">Gender:</span>
                          <span className="text-purple-300 ml-2 capitalize">{selectedSaved.user_preferences.gender}</span>
                        </div>
                      )}
                      {selectedSaved.user_preferences.hair_texture && (
                        <div className="text-sm">
                          <span className="text-gray-400">Hair Texture:</span>
                          <span className="text-purple-300 ml-2 capitalize">{selectedSaved.user_preferences.hair_texture}</span>
                        </div>
                      )}
                      {selectedSaved.user_preferences.hair_length && (
                        <div className="text-sm">
                          <span className="text-gray-400">Hair Length:</span>
                          <span className="text-purple-300 ml-2 capitalize">{selectedSaved.user_preferences.hair_length}</span>
                        </div>
                      )}
                      {selectedSaved.user_preferences.lifestyle && (
                        <div className="text-sm">
                          <span className="text-gray-400">Lifestyle:</span>
                          <span className="text-purple-300 ml-2 capitalize">{selectedSaved.user_preferences.lifestyle}</span>
                        </div>
                      )}
                      {selectedSaved.user_preferences.hair_condition && (
                        <div className="text-sm">
                          <span className="text-gray-400">Hair Condition:</span>
                          <span className="text-purple-300 ml-2 capitalize">{selectedSaved.user_preferences.hair_condition}</span>
                        </div>
                      )}
                      {selectedSaved.user_preferences.maintenance_level && (
                        <div className="text-sm">
                          <span className="text-gray-400">Maintenance:</span>
                          <span className="text-purple-300 ml-2 capitalize">{selectedSaved.user_preferences.maintenance_level}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Saved Date */}
                <div className="bg-gray-800/50 border border-gray-600/30 rounded-xl p-4">
                  <div className="flex items-center text-gray-400 text-sm">
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>
                      Saved on {new Date(selectedSaved.saved_at).toLocaleDateString('en-US', {
                        weekday: 'long',
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric'
                      })}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default UserProfile;
