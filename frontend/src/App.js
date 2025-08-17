import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import LandingPage from './pages/LandingPage';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Dashboard from './pages/Dashboard';
import PhotoUpload from './pages/PhotoUpload';
import UserPreferences from './pages/UserPreferences';
import ProtectedRoute from './utils/ProtectedRoutes';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/upload" 
            element={
              <ProtectedRoute>
                <PhotoUpload />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/preferences" 
            element={
              <ProtectedRoute>
                <UserPreferences />
              </ProtectedRoute>
            } 
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
