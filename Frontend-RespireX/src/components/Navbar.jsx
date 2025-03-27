import { useState } from "react";
import profileIcon from "../assets/icons/profile.svg";
import logo from "../assets/images/logo.png";

export const Navbar = () => {
  const [activeTab, setActiveTab] = useState("Home");
  const [profileDropdown, setProfileDropdown] = useState(false);
  return (
    <div className="w-full flex justify-center py-5 px-10 sticky top-0">
      <nav className="w-full px-10 py-3 flex justify-between items-center rounded-xl">
        <div className="text-xl font-bold text-white flex items-center gap-1">
          <img className="w-16" src={logo} alt="" />
          RespireX
        </div>
        <div className="options flex gap-10">
          <div className="hidden md:flex space-x-10">
            {["Home", "Features", "Consultation"].map((tab) => (
              <a
                key={tab}
                href="#"
                onClick={() => setActiveTab(tab)}
                className={`text-white text-sm px-6 h-8 py-1 flex justify-center items-center text-nowrap rounded-full transition-all duration-300 ${
                  activeTab === tab
                    ? "bg-primary-blue font-semibold text-white"
                    : "hover:text-blue-600 text-black"
                }`}
              >
                {tab}
              </a>
            ))}
          </div>
          <div className="relative group">
            <button className="focus:outline-none flex justify-center items-center">
              <img
                src={profileIcon}
                alt="Profile"
                className="w-10 h-10 rounded-full"
                onClick={() => setProfileDropdown(!profileDropdown)}
              />
            </button>
            <div
              className={`absolute right-0 mt-2 bg-dark-blue text-white shadow-lg rounded-lg w-40 ${
                profileDropdown ? "block" : "hidden"
              }`}
            >
              <a href="#" className="block px-4 py-2">
                View Profile
              </a>
              <a href="#" className="block px-4 py-2">
                Logout
              </a>
            </div>
          </div>
        </div>
      </nav>
    </div>
  );
};
