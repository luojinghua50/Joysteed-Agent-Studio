"""Initialize agent-rag with default FAQ and docs knowledge bases."""
import asyncio
import httpx
import io

RAG_URL = "http://127.0.0.1:8010"

FAQ_DATA = [
    ("退换货政策.txt", "退换货政策\n\n我们提供7天无理由退换货服务。商品签收后7天内，保持原包装完好即可申请退换。退款将在审核通过后3个工作日内原路退回。"),
    ("配送时效.txt", "配送时效\n\n标准配送：3-5个工作日送达。加急配送：1-2个工作日送达（需额外付费）。偏远地区可能延迟1-2天。"),
    ("会员权益.txt", "会员权益\n\nVIP会员享受：1. 全场95折 2. 优先客服通道 3. 专属优惠券 4. 生日礼品 5. 免费加急配送。年度消费满5000元自动升级。"),
    ("支付方式.txt", "支付方式\n\n支持支付宝、微信支付、银行卡（借记卡/信用卡）、花呗分期。分期免息活动请关注首页活动页。"),
    ("发票开具.txt", "发票开具\n\n订单完成后可在\"我的订单-申请发票\"中开具电子发票。支持个人和企业发票，电子发票将在申请后24小时内发送到您的邮箱。"),
    ("账号安全.txt", "账号安全\n\n建议定期修改密码，开启手机验证码登录。如发现账号异常，请立即联系客服冻结账号。不要向他人透露验证码。"),
]

DOCS_DATA = [
    ("无线耳机使用指南.txt", "产品使用指南 - 无线耳机\n\n配对方法：1. 打开耳机盒 2. 长按配对键3秒 3. 在手机蓝牙中搜索设备 4. 点击连接。重置方法：同时按住两只耳机10秒，指示灯闪红后松开。"),
    ("智能手表使用指南.txt", "产品使用指南 - 智能手表\n\n首次使用：1. 充电至少30分钟 2. 长按右侧按钮开机 3. 下载APP扫码绑定 4. 允许权限后同步数据。常见问题：心率不准请佩戴紧贴手腕。"),
]


async def main():
    async with httpx.AsyncClient(base_url=RAG_URL, timeout=30.0) as client:
        health = await client.get("/health")
        if health.status_code != 200:
            print(f"agent-rag not ready: {health.status_code}")
            return

        # Create FAQ knowledge base
        resp = await client.post("/api/knowledge-bases", params={"name": "faq", "description": "常见问题知识库"})
        faq_kb = resp.json()
        faq_kb_id = faq_kb["id"]
        print(f"Created FAQ KB: {faq_kb_id}")

        for filename, content in FAQ_DATA:
            files = {"file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")}
            resp = await client.post(f"/api/knowledge-bases/{faq_kb_id}/documents", files=files)
            print(f"  Uploaded: {filename} -> {resp.status_code}")

        # Create docs knowledge base
        resp = await client.post("/api/knowledge-bases", params={"name": "docs", "description": "产品文档知识库"})
        docs_kb = resp.json()
        docs_kb_id = docs_kb["id"]
        print(f"Created Docs KB: {docs_kb_id}")

        for filename, content in DOCS_DATA:
            files = {"file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")}
            resp = await client.post(f"/api/knowledge-bases/{docs_kb_id}/documents", files=files)
            print(f"  Uploaded: {filename} -> {resp.status_code}")

        print("Done! Knowledge bases initialized.")


if __name__ == "__main__":
    asyncio.run(main())
