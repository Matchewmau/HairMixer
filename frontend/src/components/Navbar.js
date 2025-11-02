import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';

const Navbar = ({ transparent = true, showBackButton = false, backPath = '/', user = null, onLogout = null }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isScrolled, setIsScrolled] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      setIsScrolled(scrollTop > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showUserMenu && !event.target.closest('.user-menu')) {
        setShowUserMenu(false);
      }
      // Close mobile menu when clicking outside
      if (mobileMenuOpen && !event.target.closest('#navbar-menu') && !event.target.closest('[aria-controls="navbar-menu"]')) {
        setMobileMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showUserMenu, mobileMenuOpen]);

  // Determine if navbar should use dark styling (better contrast)
  const useDarkStyling = !transparent || isScrolled;

  // Helper function to check if a nav link is active
  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      useDarkStyling
        ? 'bg-gray-900/95 backdrop-blur-md shadow-lg border-b border-gray-700/50' 
        : 'bg-gray-900/30 backdrop-blur-md border-b border-gray-700/30'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left side - Back button or Logo */}
          <div className="flex items-center">
            {showBackButton ? (
              <button
                type="button"
                onClick={() => navigate(backPath)}
                className="p-2 rounded-full backdrop-blur-sm transition duration-300 text-white hover:text-gray-200 hover:bg-white/10"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
            ) : (
              <Link to="/" className="text-2xl font-bold transition duration-300 text-white hover:text-gray-200">
                HairMixer
              </Link>
            )}
          </div>

          {/* Right side - Auth buttons (desktop only) and Mobile menu button */}
          <div className="flex items-center md:order-2 space-x-3">
            {user ? (
              // Authenticated user content (desktop only - hidden on mobile)
              <div className="relative user-menu hidden md:block">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition duration-300 backdrop-blur-sm text-white hover:text-gray-200 hover:bg-white/10"
                >
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-blue-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
                    {user?.firstName?.charAt(0) || user?.email?.charAt(0) || 'U'}
                  </div>
                  <span>{user?.firstName || 'User'}</span>
                  <svg className={`w-4 h-4 transition-transform duration-200 ${showUserMenu ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {/* User Dropdown Menu (Desktop) */}
                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-48 rounded-md shadow-lg backdrop-blur-md border bg-gray-800/95 border-gray-600/50 z-50">
                    <div className="py-1">
                      <Link
                        to="/profile"
                        onClick={() => setShowUserMenu(false)}
                        className="flex items-center px-4 py-2 text-sm transition duration-300 text-gray-300 hover:text-white hover:bg-white/10"
                      >
                        <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                        Profile
                      </Link>
                      <div className="border-t my-1 border-gray-600/50"></div>
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          onLogout();
                        }}
                        className="flex items-center w-full px-4 py-2 text-sm transition duration-300 text-red-400 hover:text-red-300 hover:bg-red-900/20"
                      >
                        <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                        </svg>
                        Logout
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              // Guest user content (desktop only, hidden on mobile)
              <>
                <Link
                  to="/login"
                  className="hidden md:inline-block px-4 py-2 rounded-md text-sm font-medium transition duration-300 backdrop-blur-sm text-white hover:text-gray-200 border border-gray-600/50 hover:bg-white/10"
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className="hidden md:inline-block px-4 py-2 rounded-md text-sm font-medium transition duration-300 backdrop-blur-sm bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-700 hover:to-blue-700 border border-purple-500/30"
                >
                  Sign Up
                </Link>
              </>
            )}

            {/* Mobile menu button */}
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="inline-flex items-center p-2 w-10 h-10 justify-center text-sm rounded-lg md:hidden backdrop-blur-sm transition duration-300 text-white hover:text-gray-200 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-600"
              aria-controls="navbar-menu"
              aria-expanded={mobileMenuOpen}
            >
              <span className="sr-only">Open main menu</span>
              <svg className="w-5 h-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 17 14">
                <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M1 1h15M1 7h15M1 13h15"/>
              </svg>
            </button>
          </div>

          {/* Center - Navigation Links (Desktop) */}
          <div className="items-center justify-between hidden w-full md:flex md:w-auto md:order-1">
            <ul className="flex flex-row space-x-4">
              <li>
                <Link
                  to="/"
                  className={`block px-3 py-2 rounded-md text-sm font-medium transition duration-300 ${
                    isActive('/')
                      ? 'bg-purple-600 text-white'
                      : 'text-white hover:text-gray-200 hover:bg-white/10'
                  }`}
                  aria-current={isActive('/') ? 'page' : undefined}
                >
                  Dashboard
                </Link>
              </li>
              <li>
                <Link
                  to="/discover"
                  className={`block px-3 py-2 rounded-md text-sm font-medium transition duration-300 ${
                    isActive('/discover')
                      ? 'bg-purple-600 text-white'
                      : 'text-white hover:text-gray-200 hover:bg-white/10'
                  }`}
                >
                  Discover
                </Link>
              </li>
              <li>
                <Link
                  to="/dashboard"
                  className={`block px-3 py-2 rounded-md text-sm font-medium transition duration-300 ${
                    isActive('/dashboard')
                      ? 'bg-purple-600 text-white'
                      : 'text-white hover:text-gray-200 hover:bg-white/10'
                  }`}
                >
                  Analyze
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Mobile Menu */}
        <div className={`${mobileMenuOpen ? 'block' : 'hidden'} md:hidden`} id="navbar-menu">
          <ul className="flex flex-col font-medium p-4 mt-4 rounded-lg border bg-black border-gray-700">
            <li>
              <Link
                to="/"
                onClick={() => setMobileMenuOpen(false)}
                className={`block py-2 px-3 rounded-md transition duration-300 ${
                  isActive('/')
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
                aria-current={isActive('/') ? 'page' : undefined}
              >
                Dashboard
              </Link>
            </li>
            <li>
              <Link
                to="/discover"
                onClick={() => setMobileMenuOpen(false)}
                className={`block py-2 px-3 rounded-md transition duration-300 ${
                  isActive('/discover')
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                Discover
              </Link>
            </li>
            <li>
              <Link
                to="/dashboard"
                onClick={() => setMobileMenuOpen(false)}
                className={`block py-2 px-3 rounded-md transition duration-300 ${
                  isActive('/dashboard')
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                Analyze
              </Link>
            </li>
            
            {user ? (
              // User Profile and Logout (for authenticated users on mobile)
              <>
                <li className="mt-3 pt-3 border-t border-gray-700">
                  <Link
                    to="/profile"
                    onClick={() => setMobileMenuOpen(false)}
                    className="flex items-center py-2 px-3 rounded-md text-gray-300 hover:bg-gray-700 hover:text-white transition duration-300"
                  >
                    <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    Profile
                  </Link>
                </li>
                <li className="mt-2">
                  <button
                    onClick={() => {
                      setMobileMenuOpen(false);
                      onLogout();
                    }}
                    className="flex items-center w-full py-2 px-3 rounded-md text-red-400 hover:bg-red-900/20 hover:text-red-300 transition duration-300"
                  >
                    <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                    </svg>
                    Logout
                  </button>
                </li>
              </>
            ) : (
              // Login and Sign Up buttons (for guest users on mobile)
              <>
                <li className="mt-3 pt-3 border-t border-gray-700">
                  <Link
                    to="/login"
                    onClick={() => setMobileMenuOpen(false)}
                    className="block py-2 px-3 text-center rounded-md border border-gray-600 text-gray-300 hover:bg-gray-700 hover:text-white transition duration-300"
                  >
                    Login
                  </Link>
                </li>
                <li className="mt-2">
                  <Link
                    to="/signup"
                    onClick={() => setMobileMenuOpen(false)}
                    className="block py-2 px-3 text-center rounded-md bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-700 hover:to-blue-700 transition duration-300"
                  >
                    Sign Up
                  </Link>
                </li>
              </>
            )}
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
