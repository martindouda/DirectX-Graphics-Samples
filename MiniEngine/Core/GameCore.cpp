#include "pch.h"
#include "GameCore.h"
#include "GraphicsCore.h"
#include "SystemTime.h"
#include "GameInput.h"
#include "BufferManager.h"
#include "CommandContext.h"
#include "PostEffects.h"
#include "Display.h"
#include "Util/CommandLineArg.h"
#include <shellapi.h>

// ImGui Includes
#include "imgui/imgui.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx12.h"

#pragma comment(lib, "runtimeobject.lib") 

// Forward declare the message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace GameCore
{
    using namespace Graphics;
    bool gIsSupending = false;
    HWND g_hWnd = nullptr;

    DescriptorHeap g_ImguiDescriptorHeap;

    void InitializeApplication(IGameApp& game)
    {
        int argc = 0;
        LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        CommandLineArgs::Initialize(argc, argv);

        Graphics::Initialize(game.RequiresRaytracingSupport());
        SystemTime::Initialize();
        GameInput::Initialize();
        EngineTuning::Initialize();

        // --- ImGui Initialization ---
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        // Fix for the builder nullptr: Force font atlas build before backend init
        ImGuiIO& io = ImGui::GetIO();
        unsigned char* tex_pixels = nullptr;
        int tex_w, tex_h;
        io.Fonts->GetTexDataAsRGBA32(&tex_pixels, &tex_w, &tex_h);

        ImGui_ImplWin32_Init(g_hWnd);

        g_ImguiDescriptorHeap.Create(L"ImGui Font Heap", D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 10);

		ID3D12DescriptorHeap* imguiHeapPointer = g_ImguiDescriptorHeap.GetHeapPointer();
        ImGui_ImplDX12_Init(g_Device, 3,
            DXGI_FORMAT_R11G11B10_FLOAT, // Default MiniEngine HDR Scene Color format
            imguiHeapPointer,
            imguiHeapPointer->GetCPUDescriptorHandleForHeapStart(),
            imguiHeapPointer->GetGPUDescriptorHandleForHeapStart());

        game.Startup();
    }

    void TerminateApplication(IGameApp& game)
    {
        g_CommandManager.IdleGPU();

        // --- ImGui Shutdown ---
        ImGui_ImplDX12_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();

        game.Cleanup();
        GameInput::Shutdown();
    }

    bool UpdateApplication(IGameApp& game)
    {
        EngineProfiling::Update();

        float DeltaTime = Graphics::GetFrameTime();

        // Start the ImGui frame
        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();


        GameInput::Update(DeltaTime);
        EngineTuning::Update(DeltaTime);

        game.Update(DeltaTime);
        game.RenderScene();

        PostEffects::Render();

        // Miniengine UI
        GraphicsContext& UiContext = GraphicsContext::Begin(L"Render UI");
        UiContext.TransitionResource(g_OverlayBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
        UiContext.ClearColor(g_OverlayBuffer);
        UiContext.SetRenderTarget(g_OverlayBuffer.GetRTV());
        UiContext.SetViewportAndScissor(0, 0, g_OverlayBuffer.GetWidth(), g_OverlayBuffer.GetHeight());
        
        game.RenderUI(UiContext);

        UiContext.SetRenderTarget(g_OverlayBuffer.GetRTV());
        UiContext.SetViewportAndScissor(0, 0, g_OverlayBuffer.GetWidth(), g_OverlayBuffer.GetHeight());
        EngineTuning::Display(UiContext, 10.0f, 40.0f, 1900.0f, 1040.0f);

        UiContext.Finish();

        // ImGui UI
		GraphicsContext& ImGuiContext = GraphicsContext::Begin(L"ImGui Render");
        ImGuiContext.TransitionResource(g_OverlayBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
        ImGuiContext.SetRenderTarget(g_OverlayBuffer.GetRTV());
        ImGuiContext.SetViewportAndScissor(0, 0, g_OverlayBuffer.GetWidth(), g_OverlayBuffer.GetHeight());

		game.RenderImGui(ImGuiContext);
		ID3D12DescriptorHeap* imguiHeapPointer = g_ImguiDescriptorHeap.GetHeapPointer();
        ImGuiContext.GetCommandList()->SetDescriptorHeaps(1, &imguiHeapPointer);
        ImGui::Render();
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), ImGuiContext.GetCommandList());

        ImGuiContext.SetRenderTarget(g_OverlayBuffer.GetRTV());
        ImGuiContext.SetViewportAndScissor(0, 0, g_OverlayBuffer.GetWidth(), g_OverlayBuffer.GetHeight());
        EngineTuning::Display(ImGuiContext, 10.0f, 40.0f, 1900.0f, 1040.0f);

        ImGuiContext.Finish();


        Display::Present();

        return !game.IsDone();
    }

    bool IGameApp::IsDone(void)
    {
        return GameInput::IsFirstPressed(GameInput::kKey_escape);
    }

    LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

    int RunApplication(IGameApp& app, const wchar_t* className, HINSTANCE hInst, int nCmdShow)
    {
        if (!XMVerifyCPUSupport())
            return 1;

        Microsoft::WRL::Wrappers::RoInitializeWrapper InitializeWinRT(RO_INIT_MULTITHREADED);
        ASSERT_SUCCEEDED(InitializeWinRT);

        // Register class
        WNDCLASSEX wcex;
        wcex.cbSize = sizeof(WNDCLASSEX);
        wcex.style = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc = WndProc;
        wcex.cbClsExtra = 0;
        wcex.cbWndExtra = 0;
        wcex.hInstance = hInst;
        wcex.hIcon = LoadIcon(hInst, IDI_APPLICATION);
        wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wcex.lpszMenuName = nullptr;
        wcex.lpszClassName = className;
        wcex.hIconSm = LoadIcon(hInst, IDI_APPLICATION);
        ASSERT(0 != RegisterClassEx(&wcex), "Unable to register a window");

        // Create window
        RECT rc = { 0, 0, (LONG)g_DisplayWidth, (LONG)g_DisplayHeight };
        AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

        g_hWnd = CreateWindow(className, className, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
            rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, hInst, nullptr);

        ASSERT(g_hWnd != 0);

        InitializeApplication(app);

        ShowWindow(g_hWnd, nCmdShow/*SW_SHOWDEFAULT*/);

        do
        {
            MSG msg = {};
            bool done = false;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);

                if (msg.message == WM_QUIT)
                    done = true;
            }

            if (done)
                break;
        } while (UpdateApplication(app));	// Returns false to quit loop

        TerminateApplication(app);
        Graphics::Shutdown();
        return 0;
    }

    LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        // Let ImGui intercept mouse/keyboard messages
        if (ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam))
            return true;

        switch (message)
        {
        case WM_SIZE:
            Display::Resize((UINT)(UINT64)lParam & 0xFFFF, (UINT)(UINT64)lParam >> 16);
            break;

        case WM_DESTROY:
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }

        return 0;
    }
}