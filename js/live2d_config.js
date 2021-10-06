L2Dwidget.init({
    model: {
        scale: 1,
        hHeadPos: 0.5,
        vHeadPos: 0.618,
        // jsonPath: 'https://cdn.jsdelivr.net/npm/live2d-widget-model-hibiki@1.0.5/assets/hibiki.model.json',       // xxx.model.json 的路径
        // jsonPath: 'https://cdn.jsdelivr.net/npm/live2d-widget-model-epsilon2_1@1.0.5/assets/Epsilon2.1.model.json',
        // jsonPath: 'https://cdn.jsdelivr.net/npm/live2d-widget-model-gf@1.0.5/assets/Gantzert_Felixander.model.json',
        // jsonPath: 'https://cdn.jsdelivr.net/npm/live2d-widget-model-shizuku@1.0.5/assets/shizuku.model.json',
        jsonPath: 'https://cdn.jsdelivr.net/npm/live2d-widget-model-hijiki@1.0.5/assets/hijiki.model.json',  // 黑猫
        // jsonPath: 'https://cdn.jsdelivr.net/npm/live2d-widget-model-haruto@1.0.5/assets/haruto.model.json',
    },
    display: {
        superSample: 2,     // 超采样等级
        width: 200,         // canvas的宽度
        height: 300,        // canvas的高度
        position: 'left',   // 显示位置：左或右
        hOffset: 0,         // canvas水平偏移
        vOffset: 0,         // canvas垂直偏移
    },
    mobile: {
        show: true,         // 是否在移动设备上显示
        scale: 1,           // 移动设备上的缩放
        motion: true,       // 移动设备是否开启重力感应
    },
    react: {
        opacityDefault: 1,  // 默认透明度
        opacityOnHover: 1,  // 鼠标移上透明度
    },
 });
