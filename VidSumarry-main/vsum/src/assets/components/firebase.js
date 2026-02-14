// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDiKIm88l6yKJlbIdI6U7155VwYMytFSHU",
  authDomain: "vidsum-login.firebaseapp.com",
  projectId: "vidsum-login",
  storageBucket: "vidsum-login.firebasestorage.app",
  messagingSenderId: "796802546012",
  appId: "1:796802546012:web:427de062da7b6eddf3a999"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);