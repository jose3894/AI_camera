const { app, BrowserWindow, Menu, ipcMain } = require('electron')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow
let newWebCamWindow
let newAboutWindow

app.on('ready', () => {
    mainWindow = new BrowserWindow({icon: 'assets/img/lens.png'});
    mainWindow.loadFile('index.html')
    const mainMenu = Menu.buildFromTemplate(templateMenu)
    Menu.setApplicationMenu(mainMenu)
    mainWindow.webContents.openDevTools()
    mainWindow.on('closed', () => {
        app.quit();
    })
});

function createNewWebCamWindow(){
    newWebCamWindow = new BrowserWindow({
        width: 400,
        height: 300,
        title: 'Madrid'
    });
    newWebCamWindow.loadFile('webcam.html')
    newWebCamWindow.on('closed', () => {
        newWebCamWindow = null
    })
}

function createAboutWindow(){
    newAboutWindow = new BrowserWindow({
        width: 400,
        height: 300,
        title: 'About'
    });
    newAboutWindow.loadFile('about.html')
    newAboutWindow.on('closed', () => {
        newAboutWindow = null
    })
}

ipcMain.on('aboutWindows:new', (e, active) => {
    createAboutWindow();
});

const templateMenu = [
    {
        label: 'File',
        submenu: [
            {
                label: 'Madrid',
                accelerator: 'Ctrl+M',
                click(){
                    createNewWebCamWindow();
                }

            },
            {
                label: 'Exit',
                accelerator: 'Ctrl+Q',
                click(){
                    app.quit();
                }
            }
        ]
    },
    {
        label: 'Help',
        submenu: [
            {
                label: 'About',
                click(){
                    createAboutWindow();
                }
            }
        ]
    }
];
/*
function createWindow () {
  // Create the browser window.
  win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true
    }
  })

  // and load the index.html of the app.
  win.loadFile('index.html')
  //win.loadURL('http://192.168.1.12:5000')

  // Open the DevTools.
  //win.webContents.openDevTools()

  // Emitted when the window is closed.
  win.on('closed', () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    win = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (win === null) {
    createWindow()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
*/