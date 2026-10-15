# USER'S MANUAL

## HairMixer

**AI-Powered Hairstyle Recommendation System**

> Complete end-user guide: account setup, photo upload, preference form, top-10 recommendations, virtual try-on, and saved results.

---

User's Manual

---

## TABLE OF CONTENTS

| Section                                   | Page |
| ----------------------------------------- | ---- |
| **1.0 GENERAL INFORMATION**               |      |
| 1.1 System Overview                       | 1-1  |
| 1.2 Organization of the Manual            | 1-1  |
| **2.0 SYSTEM SUMMARY**                    |      |
| 2.1 System Configuration                  | 2-1  |
| 2.2 User Access Levels                    | 2-1  |
| 2.3 Contingencies                         | 2-2  |
| **3.0 GETTING STARTED**                   |      |
| 3.1 Creating Account and Logging In       | 3-1  |
| 3.2 System Menu                           | 3-1  |
| 3.2.1 Dashboard                           | 3-2  |
| 3.2.2 Upload Photo                        | 3-2  |
| 3.2.3 Preference Form                     | 3-3  |
| 3.2.4 Results Page                        | 3-5  |
| 3.2.5 Discover Page                       | 3-6  |
| 3.2.6 User Profile                        | 3-6  |
| 3.3 Changing Password                     | 3-7  |
| 3.4 Exit System                           | 3-7  |
| **4.0 USING THE SYSTEM**                  |      |
| 4.1 Photo Upload                          | 4-1  |
| 4.2 Preference Form                       | 4-2  |
| 4.3 Face Shape Analysis                   | 4-3  |
| 4.4 Hairstyle Recommendations             | 4-4  |
| 4.5 Hairstyle Details                     | 4-5  |
| 4.6 Virtual Try-On                        | 4-6  |
| 4.7 Saving Hairstyles                     | 4-7  |
| 4.8 Special Instructions for Best Results | 4-8  |
| **5.0 REPORTING**                         |      |
| 5.1 Report Capabilities                   | 5-1  |
| 5.2 Saving and Exporting Results          | 5-1  |

---

# 1.0 GENERAL INFORMATION

---

## 1.0 GENERAL INFORMATION

General Information section explains in general terms the system and the purpose for which it is intended.

### 1.1 System Overview

HairMixer is a web-based application that provides personalized hairstyle recommendations. The application uses artificial intelligence to analyze the user's face shape from an uploaded photo and matches it with user preferences to recommend the most suitable hairstyles.

The system offers the following capabilities:

- **Face Shape Detection** – Automatically identifies the user's face shape from a photo
- **Personalized Recommendations** – Suggests the top 10 hairstyles based on face shape and preferences
- **Virtual Try-On** – Allows users to visualize recommended hairstyles on their own photo
- **Styling Guidance** – Provides maintenance tips, product recommendations, and styling instructions
- **Save Favorites** – Enables users to bookmark hairstyles for future reference

HairMixer operates as a web application accessible through any modern internet browser. Its operational status is production ready.

### 1.2 Organization of the Manual

The user's manual consists of five sections: General Information, System Summary, Getting Started, Using The System, and Reporting.

**General Information** section explains in general terms the system and the purpose for which it is intended.

**System Summary** section provides a general overview of the system. The summary outlines the system's configuration, user access levels, and system behavior in case of any contingencies.

**Getting Started** section explains how to access HairMixer and create an account. The section presents briefly the system menu and main navigation.

**Using The System** section provides a detailed description of system functions, including photo upload, preference selection, viewing recommendations, and using the virtual try-on feature.

**Reporting** section describes in what way the recommendations and results are presented and how to save them for future reference.

---

# 2.0 SYSTEM SUMMARY

---

## 2.0 SYSTEM SUMMARY

System Summary section provides a general overview of the system. The summary outlines the system's configuration, user access levels, and system behavior in case of any contingencies.

### 2.1 System Configuration

HairMixer operates as a web application accessible through internet browsers. It is compatible with:

- Google Chrome (recommended)
- Mozilla Firefox
- Microsoft Edge
- Safari

The application can be accessed on desktop computers, laptops, tablets, and smartphones with an internet browser.

**Requirements:**

| Component            | Requirement                    |
| -------------------- | ------------------------------ |
| Internet Connection  | Required for all features      |
| Camera/Photo Access  | Required for uploading photos  |
| Minimum Screen Width | 320 pixels (mobile responsive) |
| JavaScript           | Must be enabled in browser     |

The application requires a stable internet connection in order to process photos and generate recommendations. Users can provide photos either by using their device camera to take a photo or by uploading an existing photo from their device gallery.

After accessing the website, HairMixer can be used immediately. New users must create an account to save preferences and hairstyle favorites.

### 2.2 User Access Levels

There are two user access levels:

| Access Level        | Capabilities                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Guest User**      | Can upload photos and receive recommendations during the session                                                         |
| **Registered User** | Can save favorite hairstyles, create preference profiles, view history, and access all features across multiple sessions |

Anyone can use the basic features of the application, but only registered users are able to save data and access their information across multiple sessions.

### 2.3 Contingencies

**Internet Connection Loss:**
In case of internet disconnection, the following applies:

- Photos cannot be analyzed
- Recommendations cannot be generated
- Virtual try-on feature is unavailable
- Previously saved favorites can be viewed when connection is restored

**Session Timeout:**
If the user remains inactive for an extended period, the session may expire. The user will need to log in again to access saved data.

**Photo Processing Issues:**
If a photo cannot be processed (due to quality issues or no face detected), the system will display an error message with guidance on how to provide a suitable photo.

---

# 3.0 GETTING STARTED

---

## 3.0 GETTING STARTED

Getting Started section explains how to access HairMixer and create an account. The section presents briefly the system menu and main navigation.

### 3.1 Creating Account and Logging In

HairMixer can be accessed by navigating to the website URL in any supported internet browser.

**To create a new account:**

1. Click the **"Sign Up"** button on the homepage
2. Enter your email address
3. Create a password
4. Confirm your password
5. Click **"Create Account"**

**To log in to an existing account:**

1. Click the **"Login"** button on the homepage
2. Enter your registered email address
3. Enter your password
4. Click **"Login"**

User credentials are securely stored. For security purposes, passwords are encrypted and cannot be viewed by system administrators.

### 3.2 System Menu

HairMixer consists of several main pages accessible through the navigation menu. The primary pages are: Dashboard, Upload, Results, Discover, and Profile. The navigation menu is located at the top of the screen on desktop devices and can be accessed through a menu icon on mobile devices.

#### 3.2.1 Dashboard

The Dashboard is the home page displayed after logging in. It provides quick access to:

- **Start New Analysis** – Button to begin uploading a new photo
- **Recent Recommendations** – List of recently viewed hairstyles
- **Saved Hairstyles** – Quick access to bookmarked favorites
- **Preference Profiles** – Access to saved preference sets

#### 3.2.2 Upload Photo

The Upload Photo page allows users to provide a photo for face shape analysis. There are two methods available:

**Camera Capture:**

- Click "Open Camera" to use device camera
- Take a photo when ready
- Review and confirm the photo

**Gallery Upload:**

- Click "Upload Photo" to select from device
- Browse and select a photo file
- Supported formats: JPEG, PNG, WebP

**Photo Requirements:**

For accurate face shape detection, the photo must meet these requirements:

| Requirement         | Description                                      |
| ------------------- | ------------------------------------------------ |
| **Lighting**        | Good, even lighting without harsh shadows        |
| **Face Position**   | Face clearly visible, looking straight at camera |
| **Face Visibility** | Entire face visible, hair away from forehead     |
| **Image Quality**   | Clear, not blurry                                |
| **Minimum Size**    | At least 200 x 200 pixels                        |
| **Maximum Size**    | 10 MB file size limit                            |

_Figure 1: Photo Requirements Example_

| Good photo             | Poor photo            |
| ----------------------- | --------------------- |
| Even lighting           | Dark or harsh shadows |
| Face looking forward    | Face turned to side   |
| Hair away from forehead | Hair covering face    |
| Clear image             | Blurry image          |

#### 3.2.3 Preference Form

After uploading a photo, users complete a preference form consisting of 15 questions. This information helps the system recommend hairstyles that match the user's lifestyle and preferences.

**Preference Questions:**

| #   | Question          | Options                                    |
| --- | ----------------- | ------------------------------------------ |
| 1   | Gender            | Male, Female                               |
| 2   | Hair Type         | Straight, Wavy, Curly, Coily               |
| 3   | Hair Length       | Very Short, Short, Medium, Long, Very Long |
| 4   | Hair Color        | Black, Brown, Blonde, Red, Gray, Other     |
| 5   | Volume            | Flat, Low, Medium, High, Very High         |
| 6   | Thickness         | Fine, Medium, Thick                        |
| 7   | Texture Detail    | Smooth, Normal, Frizzy, Damaged            |
| 8   | Lifestyle         | Professional, Casual, Active, Creative     |
| 9   | Maintenance Level | Low, Medium, High                          |
| 10  | Styling Time      | Low, Medium, High                          |
| 11  | Style Preference  | Natural, Polished, Edgy, Glamorous         |
| 12  | Occasions         | Work, Casual, Formal, Dating, Sports       |
| 13  | Bangs Preference  | Yes, No                                    |
| 14  | Hair Condition    | Healthy, Dry, Oily, Damaged, Colored       |
| 15  | Face Shape        | (Automatically detected from photo)        |

_Note: Face shape is determined automatically by the system based on the uploaded photo. Users do not need to select this value._

**Preference Profiles:**

Registered users can save their preference answers as named profiles (e.g., "Work Look", "Casual Style") for quick access in future sessions. This eliminates the need to answer all questions again.

#### 3.2.4 Results Page

The Results Page displays after the system has analyzed the photo and processed the preferences. This page shows:

**Face Shape Result:**

- The detected face shape (Oval, Round, Square, Heart, or Oblong)
- Confidence percentage indicating accuracy of detection

**Top 10 Recommendations:**
A list of the 10 most suitable hairstyles, displayed as cards. Each card contains:

| Element        | Description                                      |
| -------------- | ------------------------------------------------ |
| Hairstyle Name | Name of the recommended style                    |
| Preview Image  | Reference photo of the hairstyle                 |
| Match Score    | Percentage indicating how well the style matches |
| Action Buttons | View Details, Try On, Save                       |

Users can click on any hairstyle card to view detailed information or try the style virtually.

#### 3.2.5 Discover Page

The Discover Page allows users to browse all available hairstyles in the system without going through the full recommendation process. Features include:

- Browse hairstyles by category
- Filter by hair type, length, or style
- Try any hairstyle on their uploaded photo
- Save interesting hairstyles to favorites

#### 3.2.6 User Profile

The User Profile page contains account settings and saved data:

**Account Information:**

- Email address
- Password change option

**Preference Profiles:**

- List of saved preference profiles
- Create new profile
- Edit or delete existing profiles

**Saved Hairstyles:**

- List of bookmarked favorite hairstyles
- Personal notes added to each saved hairstyle

**History:**

- Record of previous analysis sessions
- Past recommendations received

### 3.3 Changing Password

To change your password:

1. Navigate to User Profile
2. Click "Change Password"
3. Enter your current password
4. Enter your new password
5. Confirm your new password
6. Click "Save Changes"

### 3.4 Exit System

To exit HairMixer, simply close the browser tab or window, or click "Logout" in the navigation menu to end your session.

---

# 4.0 USING THE SYSTEM

---

## 4.0 USING THE SYSTEM

This section provides a detailed description of system functions.

### 4.1 Photo Upload

The Photo Upload function allows users to provide a photo for face shape analysis.

**Using Device Camera:**

1. Click "Open Camera" button
2. Grant camera permission if prompted by browser
3. Position your face within the camera frame
4. Ensure good lighting and clear view of face
5. Click "Capture" to take photo
6. Review the photo
7. Click "Use This Photo" to proceed or "Retake" to try again

**Uploading From Device:**

1. Click "Upload Photo" button
2. Browse your device files
3. Select a photo file (JPEG, PNG, or WebP format)
4. Wait for upload to complete
5. Review the photo
6. Click "Analyze" to proceed

**Photo Tips for Best Results:**

- Use natural daylight or well-lit indoor areas
- Face the camera directly, looking straight ahead
- Keep hair away from your forehead to show face shape
- Remove glasses if they cause glare
- Use a plain background when possible
- Ensure the photo is clear and not blurry

If the system cannot detect a face in the photo, an error message will appear with suggestions for providing a suitable photo.

### 4.2 Preference Form

The Preference Form collects information about your hair characteristics and styling preferences.

**Completing the Form:**

1. Answer each question by selecting the most appropriate option
2. Be honest about your current hair type and condition
3. Consider your daily routine when answering lifestyle questions
4. Think about how much time you want to spend styling each day
5. Click "Submit" when all questions are answered

**Understanding the Questions:**

| Question          | How to Answer                                                   |
| ----------------- | --------------------------------------------------------------- |
| Hair Type         | Select your natural hair pattern (straight, wavy, curly, coily) |
| Hair Length       | Select your current or desired hair length                      |
| Maintenance Level | Consider how much daily styling time you have                   |
| Lifestyle         | Choose what best describes your typical day                     |
| Occasions         | Select where you most often need to look your best              |

**Saving as a Profile:**

To save your answers for future use:

1. Complete all preference questions
2. Click "Save as Profile"
3. Enter a name for the profile (e.g., "Everyday Look")
4. Click "Save"

To use a saved profile in future sessions:

1. Start a new analysis
2. At the preference form, click "Use Saved Profile"
3. Select the desired profile from the list
4. Preferences will be automatically filled in

### 4.3 Face Shape Analysis

Face shape analysis occurs automatically when a suitable photo is uploaded.

**How It Works:**

1. The system detects the face in your photo
2. Facial features are analyzed to determine proportions
3. Your face shape is classified into one of five categories
4. A confidence score indicates the accuracy of the detection

**Face Shape Categories:**

| Face Shape | Characteristics                                      |
| ---------- | ---------------------------------------------------- |
| **Oval**   | Balanced proportions, face slightly longer than wide |
| **Round**  | Equal width and length, soft features and angles     |
| **Square** | Strong jawline, forehead and jaw similar in width    |
| **Heart**  | Wider forehead, narrower chin, pointed lower face    |
| **Oblong** | Longer than wide, even proportions throughout        |

**Understanding Confidence Scores:**

| Score         | Meaning                                           |
| ------------- | ------------------------------------------------- |
| 90% or higher | Very confident, highly accurate result            |
| 70% - 89%     | Good confidence, reliable result                  |
| 50% - 69%     | Moderate confidence, result is best estimate      |
| Below 50%     | Low confidence, consider uploading a better photo |

If the confidence score is low, you may want to upload a new photo with better lighting and clearer face visibility.

### 4.4 Hairstyle Recommendations

After face shape analysis and preference submission, the system generates personalized hairstyle recommendations.

**How Recommendations Are Generated:**

1. Your face shape is combined with your preference answers
2. The system evaluates compatibility with available hairstyles
3. Hairstyles are ranked by how well they match your profile
4. The top 10 matching hairstyles are displayed

**Viewing Recommendations:**

Each recommended hairstyle is displayed as a card showing:

- **Hairstyle name** – The name of the style
- **Preview image** – A reference photo of the hairstyle
- **Match score** – How well this style suits you (higher is better)
- **Action buttons** – Options to view details, try on, or save

Click on any card to see more information about that hairstyle.

### 4.5 Hairstyle Details

The Hairstyle Details page provides comprehensive information about a selected hairstyle.

**Information Displayed:**

**Why This Works For You:**
A personalized explanation of why this hairstyle complements your face shape and hair type.

**How It Matches Your Preferences:**
Bullet points showing how the style aligns with your lifestyle, maintenance preferences, and occasions.

**Recommended Products:**
A list of hair products to achieve and maintain this look:

- Product type (e.g., styling mousse, hair serum)
- Key ingredients
- How to use the product

**Maintenance Guide:**
Step-by-step instructions for maintaining the hairstyle:

- Daily styling routine with time estimate
- Recommended washing schedule
- How often to visit a salon for trims

**Styling Tips:**
Professional techniques for achieving the best results:

- Recommended tools and heat settings
- Application techniques
- Tips for different occasions

### 4.6 Virtual Try-On

The Virtual Try-On feature allows you to see how a hairstyle would look on you.

**Using Virtual Try-On:**

1. Select a hairstyle you want to try
2. Click the "Try Hairstyle" button
3. Wait while the image is being generated (10-30 seconds)
4. View the result showing you with the new hairstyle

**What to Expect:**

The generated image will show:

- Your face with the selected hairstyle applied
- Realistic hair texture and appearance
- Original lighting and background preserved

**Saving the Try-On Image:**

1. After the image is generated, click "Save Image"
2. The image will download to your device
3. You can show this image to your stylist

**Tips for Best Results:**

- Use a photo with good lighting
- Face the camera directly
- Use a plain background if possible
- The clearer your original photo, the better the result

### 4.7 Saving Hairstyles

Registered users can save hairstyles to their favorites for future reference.

**To Save a Hairstyle:**

1. Find a hairstyle you like in recommendations or discover page
2. Click the bookmark/save icon on the hairstyle card
3. Optionally add personal notes (e.g., "Show to stylist")
4. Click "Save"

**Accessing Saved Hairstyles:**

1. Go to User Profile
2. Click "Saved Hairstyles"
3. View all your saved favorites
4. Click any saved hairstyle to view details again

**Managing Saved Hairstyles:**

- Edit notes on any saved hairstyle
- Remove hairstyles from favorites by clicking the save icon again
- Compare different saved hairstyles

### 4.8 Special Instructions for Best Results

**Photo Quality Issues:**

If face detection repeatedly fails:

1. Move to a well-lit area (natural daylight is best)
2. Face the camera directly, looking straight ahead
3. Pull hair away from your forehead
4. Remove glasses if they cause reflections
5. Use a plain, non-distracting background
6. Ensure the photo is clear, not blurry

**Preference Accuracy:**

For the most relevant recommendations:

1. Be honest about your hair type and condition
2. Realistically assess your daily styling time available
3. Consider your most common occasions, not just special events
4. Update preferences if your situation changes

**Virtual Try-On Quality:**

For better try-on images:

1. Use a high-quality original photo
2. Good lighting produces more realistic results
3. Front-facing photos work best
4. Wait for the full generation to complete before saving

---

# 5.0 REPORTING

---

## 5.0 REPORTING

Reporting section describes in what way the recommendations and results are presented and how to save them for future reference.

### 5.1 Report Capabilities

HairMixer provides several ways to view and access your results:

**Recommendation Results:**
After each analysis, the system displays your top 10 recommended hairstyles with match scores. This information includes:

- Your detected face shape
- Confidence percentage
- List of recommended hairstyles ranked by compatibility
- Detailed information for each recommended style

**Hairstyle Details:**
For each recommended hairstyle, the following information is available:

- Personalized description explaining why the style suits you
- Product recommendations
- Maintenance guide
- Styling tips

**Virtual Try-On Images:**
Generated images showing how hairstyles would look on you can be saved to your device.

**History:**
Registered users can view their previous analysis sessions, including:

- Date of each analysis
- Face shape detected in each session
- Hairstyles that were recommended

### 5.2 Saving and Exporting Results

**Saving Hairstyles:**

Registered users can save any hairstyle to their favorites:

1. Click the save/bookmark icon on any hairstyle card
2. Add optional notes
3. Access saved styles from User Profile at any time

**Downloading Try-On Images:**

To save a virtual try-on image:

1. Generate a try-on for your desired hairstyle
2. Click "Save Image" button
3. The image downloads to your device in image format
4. Show the downloaded image to your hairstylist

**Sharing Results:**

Currently, sharing directly from the application is not available. To share results:

1. Download virtual try-on images to your device
2. Take screenshots of recommendation pages if desired
3. Share downloaded files through your preferred method

**Accessing Previous Results:**

Registered users can access previous results:

1. Log in to your account
2. Go to User Profile
3. Click "History"
4. Select a previous session to view past recommendations

---

## DOCUMENT INFORMATION

| Field            | Information   |
| ---------------- | ------------- |
| Application Name | HairMixer     |
| Version          | 2.1.0         |
| Document Type    | User's Manual |
| Last Updated     | January 2026  |

---

_© 2026 HairMixer. All rights reserved._
