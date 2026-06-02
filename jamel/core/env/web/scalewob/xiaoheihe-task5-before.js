(() => {
  const user = window.AppStore.getUser();
  user.settings = user.settings || {};
  user.settings.darkMode = true;
  user.bio = "Hardcore Gamer";
  return {
    darkMode: user.settings.darkMode,
    bio: user.bio,
  };
})()
